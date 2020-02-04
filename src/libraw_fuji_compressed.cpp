/* -*- C++ -*-
 * File: libraw_fuji_compressed.cpp
 * Copyright (C) 2016 Alexey Danilchenko
 *
 * Adopted to LibRaw by Alex Tutubalin, lexa@lexa.ru
 * LibRaw Fujifilm/compressed decoder

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

// This file is #included in libraw_cxx

#ifdef _abs
#undef _abs
#undef _min
#undef _max
#endif
// twos-complement trickery
#define _abs(x) (((int)(x) ^ ((int)(x) >> 31)) - ((int)(x) >> 31))
#define _min(a, b) ((a) < (b) ? (a) : (b))
#define _max(a, b) ((a) > (b) ? (a) : (b))

struct int_pair
{
  int value1;
  int value2;
};

enum _xt_lines
{
  _R0 = 0,
  _R1,
  _R2,
  _R3,
  _R4,
  _G0 = 5,
  _G1,
  _G2,
  _G3,
  _G4,
  _G5,
  _G6,
  _G7,
  _B0 = 13,
  _B1,
  _B2,
  _B3,
  _B4,
  _ltotal = 18,
};

struct fuji_compressed_block
{
  int cur_bit;            // current bit being read (from left to right)
  int cur_pos;            // current position in a buffer
  INT64 cur_buf_offset;   // offset of this buffer in a file
  unsigned max_read_size; // Amount of data to be read
  int cur_buf_size;       // buffer size
  uchar *cur_buf;         // currently read block
  int fillbytes;          // Counter to add extra byte for block size N*16
  LibRaw_abstract_datastream *input;
  struct int_pair grad_even[3][41]; // tables of gradients
  struct int_pair grad_odd[3][41];
  ushort *linealloc;
  ushort *linebuf[_ltotal];
};

static unsigned sgetn(int n, uchar *s)
{
  unsigned result = 0;
  while (n-- > 0)
    result = (result << 8) | (*s++);
  return result;
}

// weird: this doesn't do any reads.
void LibRaw::init_fuji_compr(struct fuji_compressed_params *info)
{
  int cur_val, i;
  int8_t *qt;

  if ((libraw_internal_data.unpacker_data.fuji_block_width % 3 &&
       libraw_internal_data.unpacker_data.fuji_raw_type == 16) ||
      (libraw_internal_data.unpacker_data.fuji_block_width & 1 &&
       libraw_internal_data.unpacker_data.fuji_raw_type == 0))
    derror();

  info->q_table = (int8_t *)malloc(32768);
  merror(info->q_table, "init_fuji_compr()");

  if (libraw_internal_data.unpacker_data.fuji_raw_type == 16)
    info->line_width = (libraw_internal_data.unpacker_data.fuji_block_width * 2) / 3;
  else
    info->line_width = libraw_internal_data.unpacker_data.fuji_block_width >> 1;

  info->q_point[0] = 0;
  info->q_point[1] = 0x12;  //    0b 00010010
  info->q_point[2] = 0x43;  // == 0b 01000011
  info->q_point[3] = 0x114; // == 0b100010100
  // This is 2^14-1 for 14-bit numbers = 16383
  info->q_point[4] = (1 << libraw_internal_data.unpacker_data.fuji_bits) - 1;
  // 64 = 2^6
  info->min_value = 0x40;

  cur_val = -info->q_point[4]; // Theoretically -16383
  // Sweep from -16383 to +16383, and the q_table has enough space for that.
  // so it will be -4... -3... -2... -1.... 0...
  // it will be hard and important to get this right.
  for (qt = info->q_table; cur_val <= info->q_point[4]; ++qt, ++cur_val)
  {
    //
    if (cur_val <= -info->q_point[3])
      *qt = -4;
    else if (cur_val <= -info->q_point[2])
      *qt = -3;
    else if (cur_val <= -info->q_point[1])
      *qt = -2;
    else if (cur_val < 0)
      *qt = -1;
    else if (cur_val == 0)
      *qt = 0;
    else if (cur_val < info->q_point[1])
      *qt = 1;
    else if (cur_val < info->q_point[2])
      *qt = 2;
    else if (cur_val < info->q_point[3])
      *qt = 3;
    else
      *qt = 4;
  }

  // populting gradients
  // 14 bit, 2^14-1
  if (info->q_point[4] == 0x3FFF)
  {
    info->total_values = 0x4000;
    info->raw_bits = 14;
    info->max_bits = 56;
    info->maxDiff = 256;
  }
  // 12 bit, 2^12 - 1
  else if (info->q_point[4] == 0xFFF)
  {
    info->total_values = 4096;
    info->raw_bits = 12;
    info->max_bits = 48;
    info->maxDiff = 64;
  }
  else
    derror();
}

#define XTRANS_BUF_SIZE 0x10000

static inline void fuji_fill_buffer(struct fuji_compressed_block *info)
{
  // This basically provides a way of windowing information from the disk.
  // So, you call this to ensure there's enough info in the buffer. When it runs
  // out, it reads another 64k and says "hey this is pointed at a new offset" (cur_buf_offset).
  if (info->cur_pos >= info->cur_buf_size)
  {
    info->cur_pos = 0;
    info->cur_buf_offset += info->cur_buf_size;
#ifdef LIBRAW_USE_OPENMP
#pragma omp critical
#endif
    {
#ifndef LIBRAW_USE_OPENMP
      info->input->lock();
#endif
      info->input->seek(info->cur_buf_offset, SEEK_SET);
      auto byte_count = _min(info->max_read_size, XTRANS_BUF_SIZE);
      //printf("Buffer now points between %d and %d\n", info->cur_buf_offset, info->cur_buf_offset + byte_count);
      info->cur_buf_size = info->input->read(info->cur_buf, 1, byte_count);
#ifndef LIBRAW_USE_OPENMP
      info->input->unlock();
#endif
      // I wonder what this is for? I haven't run into it.
      // I think it's probably like "if we failed to read but need data, just extend it as 0s"
      if (info->cur_buf_size < 1) // nothing read
      {
        // fillbytes is either 1 or 0.
        printf("Fillbytes %d\n", info->fillbytes);
        if (info->fillbytes > 0)
        {
          // So if fillbytes is set, this should always be 1.
          int ls = _max(1, _min(info->fillbytes, XTRANS_BUF_SIZE));
          printf("ls %d\n", ls);
          memset(info->cur_buf, 0, ls);
          info->fillbytes -= ls;
        }
        else
          throw LIBRAW_EXCEPTION_IO_EOF;
      }
      info->max_read_size -= info->cur_buf_size;
    }
  }
}

void LibRaw::init_fuji_block(struct fuji_compressed_block *info, const struct fuji_compressed_params *params,
                             INT64 raw_offset, unsigned dsize)
{
  // dsize is the size of the block.
  // Space for (line_width + 2) * 18 shorts
  // This "linealloc" is just for memory management; it's not used directly.
  info->linealloc = (ushort *)calloc(sizeof(ushort), _ltotal * (params->line_width + 2));
  merror(info->linealloc, "init_fuji_block()");

  // oh I think this is the file size
  INT64 fsize = libraw_internal_data.internal_data.input->size();
  //printf("fsize = %lu \n", fsize);
  // Either read to the end of the file, or the entire block.
  info->max_read_size = _min(unsigned(fsize - raw_offset), dsize); // Data size may be incorrect?
  info->fillbytes = 1;

  info->input = libraw_internal_data.internal_data.input;
  // Store pointers, so you can call
  // (info->linebuf[color])[x]
  // and get a direct index into the read line.
  info->linebuf[_R0] = info->linealloc;
  for (int i = _R1; i <= _B4; i++)
    // there's an extra 2 bytes spacing after every 512-byte line.
    info->linebuf[i] = info->linebuf[i - 1] + params->line_width + 2;

  // init buffer
  // XTRANS_BUF_SIZE == 65536, so this is 64kB
  info->cur_buf = (uchar *)malloc(XTRANS_BUF_SIZE);
  merror(info->cur_buf, "init_fuji_block()");
  info->cur_bit = 0;
  info->cur_pos = 0;
  info->cur_buf_offset = raw_offset;
  // I don't know what this does yet.
  // Each gradients has 3 colors and 41 values per color, but they're all init'd to maxDiff and 1.
  // maxDiff = 256 because this is a 14bit system.
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 41; i++)
    {
      info->grad_even[j][i].value1 = params->maxDiff;
      info->grad_even[j][i].value2 = 1;
      info->grad_odd[j][i].value1 = params->maxDiff;
      info->grad_odd[j][i].value2 = 1;
    }

  info->cur_buf_size = 0;
  //printf("init_fuji_block doing fuji_fill_buffer\n");
  fuji_fill_buffer(info);
}

void LibRaw::copy_line_to_xtrans(struct fuji_compressed_block *info, int cur_line, int cur_block, int cur_block_width)
{
  ushort *lineBufB[3];
  ushort *lineBufG[6];
  ushort *lineBufR[3];
  ushort *line_buf;
  int index;

  // cur_block_width * cur_block + (6 * raw_width * cur_line)
  // cur_block_width = width of vertical stripe == 768
  // cur_block = [0..8)
  // cur_line = [0..673], but a "line" here is 6 pixel lines
  int offset = libraw_internal_data.unpacker_data.fuji_block_width * cur_block + 6 * imgdata.sizes.raw_width * cur_line;
  ushort *raw_block_data_start = imgdata.rawdata.raw_image + offset;
  int row_count = 0;

  for (int i = 0; i < 3; i++)
  {
    // uhhhhhhhhh this is awful. Ok:
    // lineBufR[0]  == linebuf[_R2], but starts from the first actual value, not the slop value.
    // lineBufR[1]  == linebuf[_R3], but starts from the first actual value, not the slop value.
    // lineBufR[2]  == linebuf[_R4], but starts from the first actual value, not the slop value.
    lineBufR[i] = info->linebuf[_R2 + i] + 1;
    lineBufB[i] = info->linebuf[_B2 + i] + 1;
  }
  // same as above, but there's more green.
  for (int i = 0; i < 6; i++) {
    lineBufG[i] = info->linebuf[_G2 + i] + 1;
  }

  // TODO: figure out row_count is used
  for (unsigned row_count = 0; row_count < 6; row_count++) {
    ushort *raw_block_data = raw_block_data_start + imgdata.sizes.raw_width * row_count;
    for (unsigned pixel_count = 0; pixel_count < cur_block_width; pixel_count++) {
      switch (imgdata.idata.xtrans_abs[row_count][(pixel_count % 6)])
      {
      case 0: // red
        // Oh, this is >> 1 because there's only 3 lines for R & B but there's 6
        // for G
        line_buf = lineBufR[row_count >> 1];
        break;
      case 1:  // green
      default: // to make static analyzer happy
        line_buf = lineBufG[row_count];
        break;
      case 2: // blue
        line_buf = lineBufB[row_count >> 1];
        break;
      }

      //0x7F..FE is 32-bit 0b01....10
      // pixel_count * 2 / 3 produces 0,0,1,1,2,2,3,3
      index = (((pixel_count * 2 / 3) & 0x7FFFFFFE) | ((pixel_count % 3) & 1)) + ((pixel_count % 3) >> 1);
      //auto color = imgdata.idata.xtrans_abs[row_count][(pixel_count % 6)];
      //printf("color %d, index %d\n", color, index);
      raw_block_data[pixel_count] = line_buf[index];
    }
  }
}

void LibRaw::copy_line_to_bayer(struct fuji_compressed_block *info, int cur_line, int cur_block, int cur_block_width)
{
  ushort *lineBufB[3];
  ushort *lineBufG[6];
  ushort *lineBufR[3];
  unsigned pixel_count;
  ushort *line_buf;

  int fuji_bayer[2][2];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 2; c++)
      fuji_bayer[r][c] = FC(r, c); // We'll downgrade G2 to G below

  int offset = libraw_internal_data.unpacker_data.fuji_block_width * cur_block + 6 * imgdata.sizes.raw_width * cur_line;
  ushort *raw_block_data = imgdata.rawdata.raw_image + offset;
  int row_count = 0;

  for (int i = 0; i < 3; i++)
  {
    lineBufR[i] = info->linebuf[_R2 + i] + 1;
    lineBufB[i] = info->linebuf[_B2 + i] + 1;
  }
  for (int i = 0; i < 6; i++)
    lineBufG[i] = info->linebuf[_G2 + i] + 1;

  while (row_count < 6)
  {
    pixel_count = 0;
    while (pixel_count < cur_block_width)
    {
      switch (fuji_bayer[row_count & 1][pixel_count & 1])
      {
      case 0: // red
        line_buf = lineBufR[row_count >> 1];
        break;
      case 1:  // green
      case 3:  // second green
      default: // to make static analyzer happy
        line_buf = lineBufG[row_count];
        break;
      case 2: // blue
        line_buf = lineBufB[row_count >> 1];
        break;
      }

      raw_block_data[pixel_count] = line_buf[pixel_count >> 1];
      ++pixel_count;
    }
    ++row_count;
    raw_block_data += imgdata.sizes.raw_width;
  }
}

#define fuji_quant_gradient(i, v1, v2) (9 * i->q_table[i->q_point[4] + (v1)] + i->q_table[i->q_point[4] + (v2)])

// Advances bit / byte pointers to immediately _after_ the next zero.
// count stores the number of 1 bits skipped to get to the zero.
static inline void fuji_zerobits(struct fuji_compressed_block *info, int *count)
{
  *count = 0;
  while (true)
  {
    // Read the bit in the given position (bit 0 is "leftmost")
    uchar zero = (info->cur_buf[info->cur_pos] >> (7 - info->cur_bit)) & 1;
    // wrapping increment
    info->cur_bit++;
    info->cur_bit &= 7;
    // if we wrapped around to zero
    if (!info->cur_bit)
    {
      // move one byte along and fill the buffer if we ran out
      ++info->cur_pos;
      fuji_fill_buffer(info);
    }

    // if zero, return.
    // info->cur_pos, cur_bit are positioned such that they point immediately after the first zero.
    // count now stores the number of 1s between where the pointers were, and the first zero.
    if (zero)
      break;
    ++*count;
  }
}

// sets code to the next bits_to_read bits.
// weird: how many bytes can `int` hold? I thought it was just 4 bytes.
// it's a regular stack-allocated int, so you'll corrupt the stack if you go out of bounds...
// Reads `bits_to_read` from the stream, and stores them in `code`.
static inline void fuji_read_code(struct fuji_compressed_block *info, int *data, int bits_to_read)
{
  // Probably this should say "max bits" but it don't.
  assert(bits_to_read <= 14);
  uchar bits_left = bits_to_read;
  uchar bits_left_in_byte = 8 - (info->cur_bit & 7);
  *data = 0;
  if (!bits_to_read)
    return;
  if (bits_to_read >= bits_left_in_byte)
  {
    do
    {
      // shift the int << by the number of bits we can still use
      *data <<= bits_left_in_byte;
      // we're about to read some bits, so subtract them from bits_left
      bits_left -= bits_left_in_byte;
      // load those bits from the current byte.
      // the ((1 << bits_left_in_byte) - 1) thing creates a mask of 0b1111 for
      // the bits we just loaded.
      *data |= info->cur_buf[info->cur_pos] & ((1 << bits_left_in_byte) - 1);
      // move the cursor
      ++info->cur_pos;
      // ensure the buffer is filled
      fuji_fill_buffer(info);
      bits_left_in_byte = 8;
    // this loop reads the first (possibly partial) byte, and any subsequent whole bytes
    } while (bits_left >= 8);
  }
  // if we read enough; i.e. no bits left
  if (!bits_left)
  {
    // set the current bit to whatever's left
    info->cur_bit = (8 - (bits_left_in_byte & 7)) & 7;
    return;
  }
  // load any remaining last bits; it's a partial byte.
  *data <<= bits_left;
  bits_left_in_byte -= bits_left;
  *data |= ((1 << bits_left) - 1) & ((unsigned)info->cur_buf[info->cur_pos] >> bits_left_in_byte);
  // set the bit pointer accordingly.
  info->cur_bit = (8 - (bits_left_in_byte & 7)) & 7;
}

// returns B such that (value2 << (B-1)) < value1 <= (value2 << B)
static inline int bitDiff(int value1, int value2)
{
  if (value2 < value1) {
    int decBits = 1;
    while (decBits <= 12 && (value2 << decBits) < value1){
      // do nothing
      decBits++;
    }
    return decBits;
  } else {
    return 0;
  }
}

// grads: the gradients for the given color. There's 41 of them.
static inline int fuji_decode_sample_even(struct fuji_compressed_block *info,
                                          const struct fuji_compressed_params *params, ushort *line_buf, int pos,
                                          struct int_pair *grads)
{
  int interp_val = 0;
  // ushort decBits;
  int errcnt = 0;

  // this all is for _G2 initially.
  // "One line previous" is _G1,
  // "Two lines previous" is _G0
  int sample = 0, code = 0;
  ushort *line_buf_cur = line_buf + pos;
  // from the previous line, <- 2
  int Rb = line_buf_cur[-2 - params->line_width];
  // previous line, <- 3
  int Rc = line_buf_cur[-3 - params->line_width];
  // previous line, <- 1
  int Rd = line_buf_cur[-1 - params->line_width];
  // two lines ago, <- 4
  // Ok, so this _does_ only work because everything is allocated contiguously.
  int Rf = line_buf_cur[-4 - 2 * params->line_width];
  //printf("line_buf_cur %p\n", line_buf_cur);

  int grad, gradient, diffRcRb, diffRfRb, diffRdRb;

  auto qp4 = params->q_point[4]; // (1 << 14) - 1;
  // important note about the Q table: it's 2*(qp4+1) long. The whole thing is
  // set up such that idx 0 represents val -16383, idx qp4 is the 0 point etc.
  // So, as long as Rb, Rc, Rf <= 16383, we'll be fine.
  // These values range between -4 and +4, so grad is between [-40, 40].
  // I don't know _why_ *9 but let's roll with it.
  grad = (9 * params->q_table[qp4 + (Rb - Rf)] + params->q_table[qp4 + (Rc - Rb)]);
  gradient = _abs(grad);
  // subtract + abs C,F,D from Rb
  diffRcRb = _abs(Rc - Rb);
  diffRfRb = _abs(Rf - Rb);
  diffRdRb = _abs(Rd - Rb);

  // Sort the three differences, choose the two values with the smallest deltas for the interpolation
  // e.g. if (Rc - Rb) is biggest, use Rf + Rd + 2*Rb.
  // This isn't used for _ages_ though.
  if (diffRcRb > diffRfRb && diffRcRb > diffRdRb)
    interp_val = Rf + Rd + 2 * Rb;
  else if (diffRdRb > diffRcRb && diffRdRb > diffRfRb)
    interp_val = Rf + Rc + 2 * Rb;
  else
    interp_val = Rd + Rc + 2 * Rb;

  // Skip to after the next 0 bits, return the number of 1s skipped over.
  fuji_zerobits(info, &sample);

  // max bits is 56 for 14-bit RAW, raw-bits is 14.
  // if sample < 41 (56-14-1)
  // This seems extremely weird that this number would be so high.
  // 41 consecutive ones would be represented FFFFFFFFFF[extra stuff]...
  // I wonder if it's used as a padding mechanism?
  if (sample < params->max_bits - params->raw_bits - 1)
  {
    if (sample > 14) {
      // This happens somewhat regularly, but not crazy regularly.
      // printf("Got an extremely large val for sample: %d\n", sample);
    }
    // initially, this will be 8 bits, but it grows as value1 changes.
    int decBits = bitDiff(grads[gradient].value1, grads[gradient].value2);
    if (decBits >= 8) {
      //printf("large decBits %d\n", decBits);
    }
    fuji_read_code(info, &code, decBits);
    // Add the number of 1s shifted by decBits.
    // So 'code' is now [num bits]code
    // VERIFIED: You could use |= here and it would work just as well!
    code += sample << decBits;
  }
  else
  {
    //printf("Got an extremely large val for sample: %d\n", sample);
    fuji_read_code(info, &code, params->raw_bits);
    code++;
  }

  // if code < 0 or >= 2^14, it doesn't make sense; because we're supposed to
  // have 14-bit output integers
  if (code < 0 || code >= params->total_values) {
    assert(false);
    errcnt++;
  }

  // the idea here is -- odd values to one side, even values to the other.
  if (code & 1) {
    // if it's odd
    // halve it, add one, make it negative
    // code = -(code / 2 + 1);
    code = -1 - code / 2;
  }
  else {
    code /= 2;
  }

  // WOAH the gradient adapts over time
  // Each gradient starts off as a pair of (256, 1) (for 14-bit)
  // += abs(code)
  // This increases much faster than value2.
  grads[gradient].value1 += _abs(code);
  // If grads[gradient].value2 == 64 / 0x40
  if (grads[gradient].value2 == params->min_value)
  {
    // Right shift both by 1.
    grads[gradient].value1 >>= 1;
    grads[gradient].value2 >>= 1;
  }
  // Add 1.
  grads[gradient].value2++;
  // grads[gradient].value2 counts from 1 -->  64 and then from 33 -> 64 repeatedly
  if (grad < 0) {
    interp_val = (interp_val >> 2) - code;
  } else {
    interp_val = (interp_val >> 2) + code;
  }
  // if we blew out the ends, fix it, by masking off the last bits.
  // This only works because we know total_values has exactly 1 bit set.
  // This code replaces the entire commented section below lol
  interp_val &= (params->total_values - 1);

  /*
  if (interp_val < 0) {
    interp_val += params->total_values;
  } else if (interp_val > params->q_point[4]) {
    interp_val -= params->total_values;
  }

  // Previously, this was clamping the value, but Fabian thought the previous
  // code should handle this, so he turned these into asserts.
  if (interp_val > params->q_point[4]) {
    assert(false);
  } else if (interp_val < 0) {
    assert(false);
  } */
  line_buf_cur[0] = interp_val;

  assert(errcnt == 0);
  return errcnt;
}

static inline int fuji_decode_sample_odd(struct fuji_compressed_block *info,
                                         const struct fuji_compressed_params *params, ushort *line_buf, int pos,
                                         struct int_pair *grads)
{
  int interp_val = 0;
  int errcnt = 0;

  int sample = 0, code = 0;
  ushort *line_buf_cur = line_buf + pos;
  int Ra = line_buf_cur[-1];
  int Rb = line_buf_cur[-2 - params->line_width];
  int Rc = line_buf_cur[-3 - params->line_width];
  int Rd = line_buf_cur[-1 - params->line_width];
  int Rg = line_buf_cur[1];

  int grad, gradient;

  grad = fuji_quant_gradient(params, Rb - Rc, Rc - Ra);
  gradient = _abs(grad);

  if ((Rb > Rc && Rb > Rd) || (Rb < Rc && Rb < Rd))
    interp_val = (Rg + Ra + 2 * Rb) >> 2;
  else
    interp_val = (Ra + Rg) >> 1;

  // store the number of 1s into 'sample'
  fuji_zerobits(info, &sample);

  if (sample < params->max_bits - params->raw_bits - 1)
  {
    int decBits = bitDiff(grads[gradient].value1, grads[gradient].value2);
    fuji_read_code(info, &code, decBits);
    code += sample << decBits;
  }
  else
  {
    fuji_read_code(info, &code, params->raw_bits);
    code++;
  }

  if (code < 0 || code >= params->total_values)
    errcnt++;

  if (code & 1)
    code = -1 - code / 2;
  else
    code /= 2;

  grads[gradient].value1 += _abs(code);
  if (grads[gradient].value2 == params->min_value)
  {
    grads[gradient].value1 >>= 1;
    grads[gradient].value2 >>= 1;
  }
  grads[gradient].value2++;
  if (grad < 0)
    interp_val -= code;
  else
    interp_val += code;
  if (interp_val < 0)
    interp_val += params->total_values;
  else if (interp_val > params->q_point[4])
    interp_val -= params->total_values;

  if (interp_val >= 0)
    line_buf_cur[0] = _min(interp_val, params->q_point[4]);
  else
    line_buf_cur[0] = 0;
  return errcnt;
}

// this writes back into the buffer that it reads from. :-/
// Output: line_buf[pos]
// Inputs:
// Rb = line_buf[pos - 2 - line_width]
// Rc = line_buf[pos - 3 - line_width]
// Rd = line_buf[pos - 1 - line_width]
// Rf = line_buf[pos - 4 - 2*line_width]
// line_width = 512??!???!??!?!?!?!?
// but the row is 768... don't know why the row is off.
// There is no "decode_interpolation_odd".
static void fuji_decode_interpolation_even(int line_width, ushort *line_buf, int pos)
{
  ushort *line_buf_cur = line_buf + pos;
  int Rb = line_buf_cur[-2 - line_width];
  int Rc = line_buf_cur[-3 - line_width];
  int Rd = line_buf_cur[-1 - line_width];
  int Rf = line_buf_cur[-4 - 2 * line_width];
  int diffRcRb = _abs(Rc - Rb);
  int diffRfRb = _abs(Rf - Rb);
  int diffRdRb = _abs(Rd - Rb);
  if (diffRcRb > diffRfRb && diffRcRb > diffRdRb)
    *line_buf_cur = (Rf + Rd + 2 * Rb) >> 2;
  else if (diffRdRb > diffRcRb && diffRdRb > diffRfRb)
    *line_buf_cur = (Rf + Rc + 2 * Rb) >> 2;
  else
    *line_buf_cur = (Rd + Rc + 2 * Rb) >> 2;
}

static void fuji_extend_generic(ushort *linebuf[_ltotal], int line_width, int start, int end)
{
  // e.g. R2 -> R4, i.e. 2-->4
  for (int i = start; i <= end; i++)
  {
    // linebuf[R2][0] = linebuf[R1][1]
    // linebuf[R3][0] = linebuf[R2][1]
    // linebuf[R4][0] = linebuf[R3][1]
    linebuf[i][0] = linebuf[i - 1][1];
    // linebuf[R2][last] = linebuf[R1][last-1]
    // linebuf[R3][last] = linebuf[R2][last-1]
    // linebuf[R4][last] = linebuf[R3][last-1]
    linebuf[i][line_width + 1] = linebuf[i - 1][line_width];
  }
}

static void fuji_extend_red(ushort *linebuf[_ltotal], int line_width)
{
  fuji_extend_generic(linebuf, line_width, _R2, _R4);
}

static void fuji_extend_green(ushort *linebuf[_ltotal], int line_width)
{
  fuji_extend_generic(linebuf, line_width, _G2, _G7);
}

static void fuji_extend_blue(ushort *linebuf[_ltotal], int line_width)
{
  fuji_extend_generic(linebuf, line_width, _B2, _B4);
}

void LibRaw::xtrans_decode_block(struct fuji_compressed_block *info, const struct fuji_compressed_params *params,
                                 int cur_line)
{
  int r_even_pos = 0, r_odd_pos = 1;
  int g_even_pos = 0, g_odd_pos = 1;
  int b_even_pos = 0, b_odd_pos = 1;

  int errcnt = 0;

  // TODO: what was line width, again?
  // (block_width * 2) / 3;
  // == 768 * 2 / 3 = 512... TODO how does this relate to anything though?
  const int line_width = params->line_width;

  for (int i = _R0; i <= _B4; i++)  {
    //printf("lineidx == %d, offset == %p\n", i, info->linebuf[i]);
  }

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    // do the first 4 slot just with this (g_even_pos == 0,2,4,6)
    if (g_even_pos < line_width)
    {
      // outputs just to the current pos (r_even_pos), offset into (info->linebuf[_R2] + 1)
      // Decodes the red value from the red-2 line.
      // info->linebuf[_R2] is zeros when this starts.
      fuji_decode_interpolation_even(line_width, info->linebuf[_R2] + 1, r_even_pos);
      r_even_pos += 2;
      // outputs to info->linebuf[_G2][1 + g_even_pos]
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G2] + 1, g_even_pos, info->grad_even[0]);
      g_even_pos += 2;
    }
    // if the while loop starts with g_even_pos=8, this triggers because of the
    // increment at the previous if block.
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R2] + 1, r_odd_pos, info->grad_odd[0]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G2] + 1, g_odd_pos, info->grad_odd[0]);
      g_odd_pos += 2;
    }
    // End effect here:
    // do 0,2,4,6 on R2 interpolation and G2 decode
    // then do
    // - R2 decode on 8
    // - G2 decode on 8
    // - R2 decode on 1
    // - G2 decode on 1
    // - R2 decode on 10
    // - G2 decode on 10
    // - R2 decode on 3
    // - G2 decode on 3
    // etc.
  }

  //backfill the edges
  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;

  // I'm pretty confused why these go two at a time
  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G3] + 1, g_even_pos, info->grad_even[1]);
      g_even_pos += 2;
      fuji_decode_interpolation_even(line_width, info->linebuf[_B2] + 1, b_even_pos);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G3] + 1, g_odd_pos, info->grad_odd[1]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B2] + 1, b_odd_pos, info->grad_odd[1]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      if (r_even_pos & 3)
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R3] + 1, r_even_pos, info->grad_even[2]);
      else
        fuji_decode_interpolation_even(line_width, info->linebuf[_R3] + 1, r_even_pos);
      r_even_pos += 2;
      fuji_decode_interpolation_even(line_width, info->linebuf[_G4] + 1, g_even_pos);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R3] + 1, r_odd_pos, info->grad_odd[2]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G4] + 1, g_odd_pos, info->grad_odd[2]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G5] + 1, g_even_pos, info->grad_even[0]);
      g_even_pos += 2;
      if ((b_even_pos & 3) == 2)
        fuji_decode_interpolation_even(line_width, info->linebuf[_B3] + 1, b_even_pos);
      else
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B3] + 1, b_even_pos, info->grad_even[0]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G5] + 1, g_odd_pos, info->grad_odd[0]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B3] + 1, b_odd_pos, info->grad_odd[0]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      if ((r_even_pos & 3) == 2)
        fuji_decode_interpolation_even(line_width, info->linebuf[_R4] + 1, r_even_pos);
      else
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R4] + 1, r_even_pos, info->grad_even[1]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G6] + 1, g_even_pos, info->grad_even[1]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R4] + 1, r_odd_pos, info->grad_odd[1]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G6] + 1, g_odd_pos, info->grad_odd[1]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      fuji_decode_interpolation_even(line_width, info->linebuf[_G7] + 1, g_even_pos);
      g_even_pos += 2;
      if (b_even_pos & 3)
        errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B4] + 1, b_even_pos, info->grad_even[2]);
      else
        fuji_decode_interpolation_even(line_width, info->linebuf[_B4] + 1, b_even_pos);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G7] + 1, g_odd_pos, info->grad_odd[2]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B4] + 1, b_odd_pos, info->grad_odd[2]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  if (errcnt)
    derror();
}

void LibRaw::fuji_bayer_decode_block(struct fuji_compressed_block *info, const struct fuji_compressed_params *params,
                                     int cur_line)
{
  int r_even_pos = 0, r_odd_pos = 1;
  int g_even_pos = 0, g_odd_pos = 1;
  int b_even_pos = 0, b_odd_pos = 1;

  int errcnt = 0;

  const int line_width = params->line_width;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R2] + 1, r_even_pos, info->grad_even[0]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G2] + 1, g_even_pos, info->grad_even[0]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R2] + 1, r_odd_pos, info->grad_odd[0]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G2] + 1, g_odd_pos, info->grad_odd[0]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G3] + 1, g_even_pos, info->grad_even[1]);
      g_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B2] + 1, b_even_pos, info->grad_even[1]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G3] + 1, g_odd_pos, info->grad_odd[1]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B2] + 1, b_odd_pos, info->grad_odd[1]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R3] + 1, r_even_pos, info->grad_even[2]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G4] + 1, g_even_pos, info->grad_even[2]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R3] + 1, r_odd_pos, info->grad_odd[2]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G4] + 1, g_odd_pos, info->grad_odd[2]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G5] + 1, g_even_pos, info->grad_even[0]);
      g_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B3] + 1, b_even_pos, info->grad_even[0]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G5] + 1, g_odd_pos, info->grad_odd[0]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B3] + 1, b_odd_pos, info->grad_odd[0]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  r_even_pos = 0, r_odd_pos = 1;
  g_even_pos = 0, g_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_R4] + 1, r_even_pos, info->grad_even[1]);
      r_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G6] + 1, g_even_pos, info->grad_even[1]);
      g_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_R4] + 1, r_odd_pos, info->grad_odd[1]);
      r_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G6] + 1, g_odd_pos, info->grad_odd[1]);
      g_odd_pos += 2;
    }
  }

  fuji_extend_red(info->linebuf, line_width);
  fuji_extend_green(info->linebuf, line_width);

  g_even_pos = 0, g_odd_pos = 1;
  b_even_pos = 0, b_odd_pos = 1;

  while (g_even_pos < line_width || g_odd_pos < line_width)
  {
    if (g_even_pos < line_width)
    {
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_G7] + 1, g_even_pos, info->grad_even[2]);
      g_even_pos += 2;
      errcnt += fuji_decode_sample_even(info, params, info->linebuf[_B4] + 1, b_even_pos, info->grad_even[2]);
      b_even_pos += 2;
    }
    if (g_even_pos > 8)
    {
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_G7] + 1, g_odd_pos, info->grad_odd[2]);
      g_odd_pos += 2;
      errcnt += fuji_decode_sample_odd(info, params, info->linebuf[_B4] + 1, b_odd_pos, info->grad_odd[2]);
      b_odd_pos += 2;
    }
  }

  fuji_extend_green(info->linebuf, line_width);
  fuji_extend_blue(info->linebuf, line_width);

  if (errcnt)
    derror();
}

// This runs once for each block
void LibRaw::fuji_decode_strip(const struct fuji_compressed_params *info_common, int cur_block, INT64 raw_offset,
                               unsigned dsize)
{
  // dsize is the size of the block / stripe. data-size probably.

  int cur_block_width, cur_line;
  unsigned line_size;
  struct fuji_compressed_block info;

  init_fuji_block(&info, info_common, raw_offset, dsize);
  // info->line_buf's still haven't been filled with anything by this point, I think.

  line_size = sizeof(ushort) * (info_common->line_width + 2);
  printf("line_size = %u (bytes)\n", line_size);

  // Check that all the line info is zeros
  for (int i = 0; i < line_size / 2; i++) {
    if (info.linealloc[i] != 0) {
      printf("AHHHHHHHHH\n");
    }
  }

  // this is block _size_, or rather, block size is actually block width.
  cur_block_width = libraw_internal_data.unpacker_data.fuji_block_width;

  // The last block is sometimes wonky apparently?
  if (cur_block + 1 == libraw_internal_data.unpacker_data.fuji_total_blocks)
  {
    cur_block_width = imgdata.sizes.raw_width - (libraw_internal_data.unpacker_data.fuji_block_width * cur_block);
    /* Old code, may get incorrect results on GFX50, but luckily large optical black
    cur_block_width = imgdata.sizes.raw_width % libraw_internal_data.unpacker_data.fuji_block_width;
    */
  }

  struct i_pair
  {
    int a, b;
  };
  const i_pair mtable[6] = {{_R0, _R3}, {_R1, _R4}, {_G0, _G6}, {_G1, _G7}, {_B0, _B3}, {_B1, _B4}},
               ztable[3] = {{_R2, 3}, {_G2, 6}, {_B2, 3}};
  // for 0 --  673
  // So, each strip contains (block_width * raw_height) u16s == 3_101_184
  // Each 'line' is 3101184 / total_lines == 4608
  // So maybe xtrans_decode_block should process that?
  for (cur_line = 0; cur_line < libraw_internal_data.unpacker_data.fuji_total_lines; cur_line++)
  {
    if (libraw_internal_data.unpacker_data.fuji_raw_type == 16) {
      // by this point, all the line buffers are still zeros.
      xtrans_decode_block(&info, info_common, cur_line);
    }
    else {
      fuji_bayer_decode_block(&info, info_common, cur_line);
    }

    // copy data from line buffers and advance
    // C2 for C in {R,G,B} --> C(last) are populated by here.
    // So C0, C1 come from the last computed coeffiencients from the last row,
    // probably?
    for (int i = 0; i < 6; i++) {
      // dst / src / byte count
      memcpy(info.linebuf[mtable[i].a], info.linebuf[mtable[i].b], line_size);
    }

    // This is the output stage! I don't know how it works yet
    if (libraw_internal_data.unpacker_data.fuji_raw_type == 16)
      copy_line_to_xtrans(&info, cur_line, cur_block, cur_block_width);
    else
      copy_line_to_bayer(&info, cur_line, cur_block, cur_block_width);

    for (int i = 0; i < 3; i++)
    {
      // Here's an example for ztable[0]
      // Set R2,R3,R4 to zeros in one fell swoop (do n lines all at once)
      memset(info.linebuf[ztable[i].a], 0, ztable[i].b * line_size);
      // set R2[0] to R1[1]
      info.linebuf[ztable[i].a][0] = info.linebuf[ztable[i].a - 1][1];
      // set R2[last] to R1[last-1]
      info.linebuf[ztable[i].a][info_common->line_width + 1] = info.linebuf[ztable[i].a - 1][info_common->line_width];
    }
  }

  // release data
  free(info.linealloc);
  free(info.cur_buf);
}

void LibRaw::fuji_compressed_load_raw()
{
  struct fuji_compressed_params common_info;
  int cur_block;
  unsigned line_size, *block_sizes;
  INT64 raw_offset, *raw_block_offsets;
  // struct fuji_compressed_block info;

  init_fuji_compr(&common_info);
  line_size = sizeof(ushort) * (common_info.line_width + 2);

  // read block sizes
  block_sizes = (unsigned *)malloc(sizeof(unsigned) * libraw_internal_data.unpacker_data.fuji_total_blocks);
  merror(block_sizes, "fuji_compressed_load_raw()");
  raw_block_offsets = (INT64 *)malloc(sizeof(INT64) * libraw_internal_data.unpacker_data.fuji_total_blocks);
  merror(raw_block_offsets, "fuji_compressed_load_raw()");

  // raw_offset here is
  // fuji_total_blocks == h_blocks_in_row
  // 8 * sizeof(unsigned)?? == 32 bits == 4 bytes?
  // So this is 8 * 4 = 32
  raw_offset = sizeof(unsigned) * libraw_internal_data.unpacker_data.fuji_total_blocks;
  printf("raw_offset == %d, sizeof(unsigned) == %d\n", raw_offset, sizeof(unsigned));
  // if either bit 3 or 2 are set
  // I don't think this is even necessary.
  /*
  if (raw_offset & 0xC) {
    // Add a row, subtract the extra bit
    // Effectively rounds to 4 byte boundary?
    // But how could this even be a problem given that
    raw_offset += 0x10 - (raw_offset & 0xC);
  }
  */

  // Add in relation to data_offset; it's relative to the file.
  raw_offset += libraw_internal_data.unpacker_data.data_offset;
  printf("raw_offset is %d\n", raw_offset);
  libraw_internal_data.internal_data.input->seek(libraw_internal_data.unpacker_data.data_offset, SEEK_SET);
  // Read 32 bytes into the block sizes array
  libraw_internal_data.internal_data.input->read(
      block_sizes, 1, sizeof(unsigned) * libraw_internal_data.unpacker_data.fuji_total_blocks);

  // This is the start of the raw data.
  raw_block_offsets[0] = raw_offset;
  // calculating raw block offsets

  // This is a little-endian to big-endian conversion
  for (cur_block = 0; cur_block < libraw_internal_data.unpacker_data.fuji_total_blocks; cur_block++)
  {
    unsigned bsize = sgetn(4, (uchar *)(block_sizes + cur_block));
    block_sizes[cur_block] = bsize;
  }

  for (cur_block = 1; cur_block < libraw_internal_data.unpacker_data.fuji_total_blocks; cur_block++) {
    // pretty straightforward, get the location of every block
    raw_block_offsets[cur_block] = raw_block_offsets[cur_block - 1] + block_sizes[cur_block - 1];
  }

  // here's the actual work.
  fuji_decode_loop(&common_info, libraw_internal_data.unpacker_data.fuji_total_blocks, raw_block_offsets, block_sizes);

  free(block_sizes);
  free(raw_block_offsets);
  free(common_info.q_table);
}

void LibRaw::fuji_decode_loop(const struct fuji_compressed_params *common_info, int count, INT64 *raw_block_offsets,
                              unsigned *block_sizes)
{
  int cur_block;
#ifdef LIBRAW_USE_OPENMP
#pragma omp parallel for private(cur_block)
#endif
  // There's 8 vertical stripes.
  for (cur_block = 0; cur_block < count; cur_block++)
  {
    fuji_decode_strip(common_info, cur_block, raw_block_offsets[cur_block], block_sizes[cur_block]);
  }
}

void LibRaw::parse_fuji_compressed_header()
{
  unsigned signature, version, h_raw_type, h_raw_bits, h_raw_height, h_raw_rounded_width, h_raw_width, h_block_size,
      h_blocks_in_row, h_total_lines;

  uchar header[16];

  libraw_internal_data.internal_data.input->seek(libraw_internal_data.unpacker_data.data_offset, SEEK_SET);
  libraw_internal_data.internal_data.input->read(header, 1, sizeof(header));

  // read all header
  signature = sgetn(2, header);
  version = header[2];
  h_raw_type = header[3];
  h_raw_bits = header[4];
  h_raw_height = sgetn(2, header + 5);
  h_raw_rounded_width = sgetn(2, header + 7);
  h_raw_width = sgetn(2, header + 9);
  h_block_size = sgetn(2, header + 11);
  h_blocks_in_row = header[13];
  h_total_lines = sgetn(2, header + 14);

  // general validation
  if (signature != 0x4953 || version != 1 || h_raw_height > 0x3000 || h_raw_height < 6 || h_raw_height % 6 ||
      h_block_size < 1 || h_raw_width > 0x3000 || h_raw_width < 0x300 || h_raw_width % 24 ||
      h_raw_rounded_width > 0x3000 || h_raw_rounded_width < h_block_size || h_raw_rounded_width % h_block_size ||
      h_raw_rounded_width - h_raw_width >= h_block_size || h_block_size != 0x300 || h_blocks_in_row > 0x10 ||
      h_blocks_in_row == 0 || h_blocks_in_row != h_raw_rounded_width / h_block_size || h_total_lines > 0x800 ||
      h_total_lines == 0 || h_total_lines != h_raw_height / 6 || (h_raw_bits != 12 && h_raw_bits != 14) ||
      (h_raw_type != 16 && h_raw_type != 0))
    return;

  // modify data
  libraw_internal_data.unpacker_data.fuji_total_lines = h_total_lines;
  libraw_internal_data.unpacker_data.fuji_total_blocks = h_blocks_in_row;
  libraw_internal_data.unpacker_data.fuji_block_width = h_block_size;
  libraw_internal_data.unpacker_data.fuji_bits = h_raw_bits;
  libraw_internal_data.unpacker_data.fuji_raw_type = h_raw_type;
  imgdata.sizes.raw_width = h_raw_width;
  imgdata.sizes.raw_height = h_raw_height;
  libraw_internal_data.unpacker_data.data_offset += 16;
  load_raw = &LibRaw::fuji_compressed_load_raw;
}

#undef _abs
#undef _min
