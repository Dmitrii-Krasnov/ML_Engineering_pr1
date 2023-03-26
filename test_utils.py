from unittest.mock import Mock

import pytest

import utils

long_source_text = "  Cluster-level shard allocation settings control allocation and rebalancing operations. \
  Disk-based shard allocation settings explains how Elasticsearch takes available disk space into account,\
   and the related settings.  Shard allocation awareness and Forced awareness control how shards can\
    be distributed across different racks or availability zones."


@pytest.mark.parametrize(
    ('txt_length', 'expected'),
    [(10, "Ok"), ]
)
def test_load_model_generator_is_called(txt_length: int, expected: str):
    """
    Check if load_model function is called

    :param txt_length (int): output length
    :param expected (str): result of model pipeline
    """
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
     (100, "Ok"),
     (1, None),
     (8, None),
     ]
)
def test_load_model_return_value(txt_length: int, expected: str):
    """
    Check, if load_model is called with valid input parameters it returns value,
    if it is called with invalid input parameters it returns None

    :param txt_length (int): output length
    :param expected (str): result of model pipeline
    """
    pipeline_mock = Mock(return_value=expected)
    utils.pipeline = pipeline_mock
    utils.cuda_device = -1

    assert utils.load_model(txt_length) == expected


@pytest.mark.parametrize(
    'txt_length',
    [-2, 0, 1, 8, ]
)
def test_load_model_small_txt_length_pipeline_not_called(txt_length: int):
    """
    Check if load_model function is called with invalid parameters and pipeline is not called

    :param txt_length (int): output length
    """
    pipeline_mock = Mock()
    utils.pipeline = pipeline_mock
    utils.load_model(txt_length)

    pipeline_mock.assert_not_called()


@pytest.mark.parametrize(
    ('generation_text', 'generation_len'),
    [(long_source_text, 5),
     ("asdsad", 15),
     ("asdsad awrwer sfsdfs sdfsdfds ", 9)]
)
def test_generate_text_with_exception(source_text: str, generation_len: int):
    """
    Check if generate_text call with invalid parameters ends with exception

    :param source_text (str):  text to be processed
    :param generation_len (int): output length
    :return:
    """
    with pytest.raises(ValueError):
        utils.generate_text(source_text, generation_len)


@pytest.mark.parametrize(
    ('generation_len', 'expected'),
    [(9, "Ok")]
)
def test_generate_text(generation_len: int, expected: str):
    """
    Check if generate_text call with valid parameters returns valid value

    :param generation_len (int): output length
     :param expected (str):  result text after generation
    :return:
    """
    load_model_mock = Mock(return_value=lambda x: [{'summary_text': expected}])
    utils.load_model = load_model_mock

    txt = utils.generate_text(long_source_text, generation_len)
    assert txt == expected


@pytest.mark.parametrize(
    ('generation_len', 'expected'),
    [(9, 9)]
)
def test_generate_text_with_realgenerator(generation_len: int, expected: int):
    """
    Check text generation with real model

    :param generation_len (int): assigned output length
    :param expected (str):  length after generation
    :return:
    """
    txt = utils.generate_text(long_source_text, generation_len)
    word_count = len(str.split(txt))
    assert 0 < word_count <= expected
