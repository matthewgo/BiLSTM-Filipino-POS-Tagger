#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

FASTTEXT=/home/matthew/Documents/python/fastText-0.2.0
VECFILE=data/fasttext-filipino/cc.tl.300.bin
DATADIR=data/filipino-pos

cat "${DATADIR}"/oov_words.txt | "${FASTTEXT}"/fasttext print-word-vectors "${VECFILE}" > "${DATADIR}"/oov_vectors.vec

