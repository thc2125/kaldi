#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
#data=/export/a15/vpanayotov/data
data=$1
# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

# you might not want to do this for interactive shells.
set -e

echoerr() { echo "$@" 1>&2; }

# download the data.  Note: we're using the 100 hour setup for
# now; later in the script we'll download more and use it to train neural
# nets.
stage=0
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  local/download_and_untar.sh $data $data_url $part
done
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."


stage=1
# download the LM resources
local/download_lm.sh $lm_url data/local/lm
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=2
# format the data as Kaldi data directories
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  # use underscore-separated names in data directories.
  local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
done
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."


stage=3
## Optional text corpus normalization and LM training
## These scripts are here primarily as a documentation of the process that has been
## used to build the LM. Most users of this recipe will NOT need/want to run
## this step. The pre-built language models and the pronunciation lexicon, as
## well as some intermediate data(e.g. the normalized text used for LM training),
## are available for download at http://www.openslr.org/11/
#local/lm/train_lm.sh $LM_CORPUS_ROOT \
#  data/local/lm/norm/tmp data/local/lm/norm/norm_texts data/local/lm

## Optional G2P training scripts.
## As the LM training scripts above, this script is intended primarily to
## document our G2P model creation process
#local/g2p/train_g2p.sh data/local/dict/cmudict data/local/lm

# when "--stage 3" option is used below we skip the G2P steps, and use the
# lexicon we have already downloaded from openslr.org/11/
local/prepare_dict.sh --stage 3 --nj 4 --cmd "$train_cmd" \
   data/local/lm data/local/lm data/local/dict_nosp
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=4
utils/prepare_lang.sh data/local/dict_nosp \
  "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=5
local/format_lms.sh --src-dir data/lang_nosp data/local/lm
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=6
# Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
  data/lang_nosp data/lang_nosp_test_tglarge
utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
  data/lang_nosp data/lang_nosp_test_fglarge
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=7
mfccdir=mfcc
# spread the mfccs over various machines, as this data-set is quite large.
if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
  mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
  utils/create_split_dir.pl /export/b{02,11,12,13}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
    $mfccdir/storage
fi
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=8
for part in dev_clean test_clean dev_other test_other train_clean_100; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 data/$part exp/make_mfcc/$part $mfccdir
  steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=9
# Make some small data subsets for early system-build stages.  Note, there are 29k
# utterances in the train_clean_100 directory which has 100 hours of data.
# For the monophone stages we select the shortest utterances, which should make it
# easier to align the data from a flat start.

utils/subset_data_dir.sh --shortest data/train_clean_100 2000 data/train_2kshort
utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=10
# train a monophone system
steps/train_mono.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
  data/train_2kshort data/lang_nosp exp/mono
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=11
# decode using the monophone model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/mono exp/mono/graph_nosp_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode.sh --nj 4 --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
      data/$test exp/mono/decode_nosp_tgsmall_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=12
steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
  data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=13
# train a first delta + delta-delta triphone system on a subset of 5000 utterances
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=14
# decode using the tri1 model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/tri1 exp/tri1/graph_nosp_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode.sh --nj 4 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgsmall \
      data/$test exp/tri1/decode_nosp_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=15
steps/align_si.sh --nj 4 --cmd "$train_cmd" \
  data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."


stage=16
# train an LDA+MLLT system.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
   data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=17
# decode using the LDA+MLLT model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/tri2b exp/tri2b/graph_nosp_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode.sh --nj 4 --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgsmall \
      data/$test exp/tri2b/decode_nosp_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      data/$test exp/tri2b/decode_nosp_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri2b/decode_nosp_{tgsmall,tglarge}_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=18
# Align a 10k utts subset using the tri2b model
steps/align_si.sh  --nj 4 --cmd "$train_cmd" --use-graphs true \
  data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali_10k
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=19
# Train tri3b, which is LDA+MLLT+SAT on 10k utts
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
  data/train_10k data/lang_nosp exp/tri2b_ali_10k exp/tri3b
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=20
# decode using the tri3b model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/tri3b exp/tri3b/graph_nosp_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
      exp/tri3b/graph_nosp_tgsmall data/$test \
      exp/tri3b/decode_nosp_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      data/$test exp/tri3b/decode_nosp_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_nosp_{tgsmall,tglarge}_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=21
# align the entire train_clean_100 subset using the tri3b model
steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
  data/train_clean_100 data/lang_nosp \
  exp/tri3b exp/tri3b_ali_clean_100
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=22
# train another LDA+MLLT+SAT system on the entire 100 hour subset
steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
  data/train_clean_100 data/lang_nosp \
  exp/tri3b_ali_clean_100 exp/tri4b
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=23
# decode using the tri4b model
(
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
    exp/tri4b exp/tri4b/graph_nosp_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
      exp/tri4b/graph_nosp_tgsmall data/$test \
      exp/tri4b/decode_nosp_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      data/$test exp/tri4b/decode_nosp_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri4b/decode_nosp_{tgsmall,tglarge}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
      data/$test exp/tri4b/decode_nosp_{tgsmall,fglarge}_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=24
# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
steps/get_prons.sh --cmd "$train_cmd" \
  data/train_clean_100 data/lang_nosp exp/tri4b
utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict_nosp \
  exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
  exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=25
utils/prepare_lang.sh data/local/dict \
  "<UNK>" data/local/lang_tmp data/lang
local/format_lms.sh --src-dir data/lang data/local/lm
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=26
utils/build_const_arpa_lm.sh \
  data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
utils/build_const_arpa_lm.sh \
  data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=27
# decode using the tri4b model with pronunciation and silence probabilities
(
  utils/mkgraph.sh \
    data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
      exp/tri4b/graph_tgsmall data/$test \
      exp/tri4b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri4b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri4b/decode_{tgsmall,tglarge}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/$test exp/tri4b/decode_{tgsmall,fglarge}_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=28
# align train_clean_100 using the tri4b model
steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
  data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=29
# if you want at this point you can train and test NN model(s) on the 100 hour
# subset
local/nnet2/run_5a_clean_100.sh
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=30
local/download_and_untar.sh $data $data_url train-clean-360
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=31
# now add the "clean-360" subset to the mix ...
local/data_prep.sh \
  $data/LibriSpeech/train-clean-360 data/train_clean_360
steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 data/train_clean_360 \
  exp/make_mfcc/train_clean_360 $mfccdir
steps/compute_cmvn_stats.sh \
  data/train_clean_360 exp/make_mfcc/train_clean_360 $mfccdir
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=32
# ... and then combine the two sets into a 460 hour one
utils/combine_data.sh \
  data/train_clean_460 data/train_clean_100 data/train_clean_360
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=33
# align the new, combined set, using the tri4b model
steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
  data/train_clean_460 data/lang exp/tri4b exp/tri4b_ali_clean_460
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=34
# create a larger SAT model, trained on the 460 hours of data.
steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
  data/train_clean_460 data/lang exp/tri4b_ali_clean_460 exp/tri5b
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=35
# decode using the tri5b model
(
  utils/mkgraph.sh data/lang_test_tgsmall \
    exp/tri5b exp/tri5b/graph_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
      exp/tri5b/graph_tgsmall data/$test \
      exp/tri5b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri5b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri5b/decode_{tgsmall,tglarge}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/$test exp/tri5b/decode_{tgsmall,fglarge}_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=36
# train a NN model on the 460 hour set
local/nnet2/run_6a_clean_460.sh
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=37
local/download_and_untar.sh $data $data_url train-other-500
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=38
# prepare the 500 hour subset.
local/data_prep.sh \
  $data/LibriSpeech/train-other-500 data/train_other_500
steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 data/train_other_500 \
  exp/make_mfcc/train_other_500 $mfccdir
steps/compute_cmvn_stats.sh \
  data/train_other_500 exp/make_mfcc/train_other_500 $mfccdir
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=39
# combine all the data
utils/combine_data.sh \
  data/train_960 data/train_clean_460 data/train_other_500
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=40
steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
  data/train_960 data/lang exp/tri5b exp/tri5b_ali_960
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=41
# train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
# as it is faster.
steps/train_quick.sh --cmd "$train_cmd" \
  7000 150000 data/train_960 data/lang exp/tri5b_ali_960 exp/tri6b
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=42
# decode using the tri6b model
(
  utils/mkgraph.sh data/lang_test_tgsmall \
    exp/tri6b exp/tri6b/graph_tgsmall
  for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" \
      exp/tri6b/graph_tgsmall data/$test exp/tri6b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri6b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri6b/decode_{tgsmall,tglarge}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/$test exp/tri6b/decode_{tgsmall,fglarge}_$test
  done
)
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=43
# this does some data-cleaning. The cleaned data should be useful when we add
# the neural net and chain systems.
local/run_cleanup_segmentation.sh
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

stage=44
# steps/cleanup/debug_lexicon.sh --remove-stress true  --nj 200 --cmd "$train_cmd" data/train_clean_100 \
#    data/lang exp/tri6b data/local/dict/lexicon.txt exp/debug_lexicon_100h

# #Perform rescoring of tri6b be means of faster-rnnlm
# #Attention: with default settings requires 4 GB of memory per rescoring job, so commenting this out by default
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 1000 -direct-order 5" \
#     --rnnlm-tag "h150-me5-1000" $data data/local/lm

# #Perform rescoring of tri6b be means of faster-rnnlm using Noise contrastive estimation
# #Note, that could be extremely slow without CUDA
# #We use smaller direct layer size so that it could be stored in GPU memory (~2Gb)
# #Suprisingly, bottleneck here is validation rather then learning
# #Therefore you can use smaller validation dataset to speed up training
# wait && local/run_rnnlm.sh \
#     --rnnlm-ver "faster-rnnlm" \
#     --rnnlm-options "-hidden 150 -direct 400 -direct-order 3 --nce 20" \
#     --rnnlm-tag "h150-me3-400-nce20" $data data/local/lm
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."


stage=45
# train nnet3 tdnn models on the entire data with data-cleaning (xent and chain)
local/chain/run_tdnn.sh # set "--stage 11" if you have already run local/nnet3/run_tdnn.sh
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."


stage=46
# The nnet3 TDNN recipe:
# local/nnet3/run_tdnn.sh # set "--stage 11" if you have already run local/chain/run_tdnn.sh

# # train models on cleaned-up data
# # we've found that this isn't helpful-- see the comments in local/run_data_cleaning.sh
# local/run_data_cleaning.sh

# # The following is the current online-nnet2 recipe, with "multi-splice".
# local/online/run_nnet2_ms.sh

# # The following is the discriminative-training continuation of the above.
# local/online/run_nnet2_ms_disc.sh

# ## The following is an older version of the online-nnet2 recipe, without "multi-splice".  It's faster
# ## to train but slightly worse.
# # local/online/run_nnet2.sh

# Wait for decodings in the background
wait
echo "6998:Completed stage: $stage..."
echoerr "6998:Completed stage: $stage..."

