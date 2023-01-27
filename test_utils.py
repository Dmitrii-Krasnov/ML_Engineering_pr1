from unittest.mock import Mock

import pytest

import utils

source_text = "  Cluster-level shard allocation settings control allocation and rebalancing operations. \
  Disk-based shard allocation settings explains how Elasticsearch takes available disk space into account,\
   and the related settings.  Shard allocation awareness and Forced awareness control how shards can\
    be distributed across different racks or availability zones."


@pytest.mark.parametrize(
    ('txt_length', 'expected'),
    [(10, "Ok"), ]
)
def test_load_model_generator_is_called(txt_length, expected):
    pipeline_mock = Mock(return_value=expected)
    utils.pipeline = pipeline_mock
    utils.cuda_device = -1
    utils.load_model(txt_length)

    pipeline_mock.assert_called_once_with("summarization",
                                          "pszemraj/long-t5-tglobal-base-16384-book-summary",
                                          max_length=txt_length,
                                          device=-1)


@pytest.mark.parametrize(
    ('txt_length', 'expected'),
    [(9, "Ok"),
     (100, "Ok")]
)
def test_load_model_return_value(txt_length, expected):
    pipeline_mock = Mock(return_value=expected)
    utils.pipeline = pipeline_mock
    utils.cuda_device = -1

    assert utils.load_model(txt_length) == expected


@pytest.mark.parametrize(
    ('txt_length', 'expected'),
    [(1, None),
     (8, None), ]
)
def test_load_model_small_txt_length_return_none(txt_length, expected):
    pipeline_mock = Mock(return_value=expected)
    utils.pipeline = pipeline_mock
    utils.cuda_device = 0

    assert utils.load_model(txt_length) == expected


@pytest.mark.parametrize(
    'txt_length',
    [-2, 0, 1, 8, ]
)
def test_load_model_small_txt_length_generator_not_called(txt_length):
    pipeline_mock = Mock()
    utils.pipeline = pipeline_mock
    utils.load_model(txt_length)

    pipeline_mock.assert_not_called()


@pytest.mark.parametrize(
    ('generation_text', 'generation_len'),
    [(source_text, 5),
     ("asdsad", 15),
     ("asdsad awrwer sfsdfs sdfsdfds ", 9)]
)
def test_generate_text_with_exception(generation_text, generation_len):
    with pytest.raises(ValueError):
        utils.generate_text(generation_text, generation_len)


@pytest.mark.parametrize(
    ('generation_len', 'expected'),
    [(9, "Ok")]
)
def test_generate_text(generation_len, expected):
    load_model_mock = Mock(return_value=lambda x: [{'summary_text': expected}])
    utils.load_model = load_model_mock

    txt = utils.generate_text(source_text, generation_len)
    assert txt == expected


@pytest.mark.parametrize(
    ('generation_len', 'expected'),
    [(9, 9)]
)
def test_generate_text_with_realgenerator(generation_len, expected):
    txt = utils.generate_text(source_text, generation_len)
    word_count = len(str.split(txt))
    assert 0 < word_count <= expected
