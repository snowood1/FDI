import nltk
import re
# =========== For debugging  ================  #

def _apply_masked_spans_with_color(doc, masked_spans, mask_type_to_substitution, add_colors):
    if None in doc:
      raise ValueError()

    color_text = doc[:]
    context = doc[:]
    answers = []

    for (span_type, span_off, span_len) in masked_spans:
      if span_len == 0:
        continue

      if span_off >= len(context):
        raise ValueError()

      answers.append((span_type, context[span_off:span_off + span_len]))
      context[span_off:span_off + span_len] = [None] * span_len
      color_text[span_off:span_off + span_len] = [None] * span_len

    for (_, span) in answers:
      if None in span:
        raise ValueError('Overlapping mask detected')

    for i, (span_type, span_off, span_len) in enumerate(masked_spans):
      offset = context.index(None)
      assert all([i is None for i in context[offset:offset + span_len]])
      del context[offset:offset + span_len]
      substitution = mask_type_to_substitution[span_type]

      if type(substitution) == list:
        context[offset:offset] = substitution
      else:
        context.insert(offset, substitution)

    if add_colors:
      for i, (span_type, span_off, span_len) in enumerate(masked_spans):
        offset = color_text.index(None)
        assert all([i is None for i in color_text[offset:offset + span_len]])
        del color_text[offset:offset + span_len]
        color_span = list(add_colors['color_code']) + doc[span_off:span_off + span_len] + list(add_colors['end_code'])
        if type(color_span) == list:
          color_text[offset:offset] = color_span
        else:
          color_text.insert(offset, color_span)
    else:
      color_text = doc

    assert None not in context
    assert None not in color_text

    return context, answers, color_text


def apply_masked_spans_with_color(doc_str_or_token_list, masked_spans,
                       mask_type_to_substitution, add_colors):
  if type(doc_str_or_token_list) == str:
    context, answers, color_text = _apply_masked_spans_with_color(
      list(doc_str_or_token_list), masked_spans,
      {k: list(v)
       for k, v in mask_type_to_substitution.items()}, add_colors)
    context = ''.join(context)
    answers = [(t, ''.join(s)) for t, s in answers]
    color_text = ''.join(color_text)
    return context, answers, color_text
  elif type(doc_str_or_token_list) == list:
    return _apply_masked_spans_with_color(doc_str_or_token_list, masked_spans,
                               mask_type_to_substitution, add_colors)
  else:
    raise ValueError()


class COLORS:
    End = "\033[0m"
    Bold = "\033[1m"
    Dim = "\033[2m"
    Underlined = "\033[4m"
    Blink = "\033[5m"
    Reverse = "\033[7m"
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"
    BackgroundBlack = "\033[40m"
    BackgroundRed = "\033[41m"
    BackgroundGreen = "\033[42m"
    BackgroundYellow = "\033[43m"
    BackgroundBlue = "\033[44m"
    BackgroundMagenta = "\033[45m"
    BackgroundCyan = "\033[46m"
    BackgroundLightGray = "\033[47m"
    BackgroundDarkGray = "\033[100m"
    BackgroundLightRed = "\033[101m"
    BackgroundLightGreen = "\033[102m"
    BackgroundLightYellow = "\033[103m"
    BackgroundLightBlue = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan = "\033[106m"
    BackgroundWhite = "\033[107m"

def get_color_fonts(colors, tokenizer):
    if type(colors) == str:
        colors = [colors]
    code = ''.join(COLORS.__dict__[c] for c in colors)

    idx = tokenizer.encode(code)

    end_idx = tokenizer.encode(COLORS.__dict__['End'])

    return {'color_code': code, 'color_idx': idx, 'end_code': COLORS.End, 'end_idx': end_idx}


def remove_color_fonts(text, fonts):
    text = text.replace(fonts['color_code'], ' __COLOR__ ')
    text = text.replace(fonts['end_code'], ' __COLOR__ ')
    text = re.sub("\s*__COLOR__\s*", ' ', text)
    return text

def get_sentences_offset(doc):
    paragraphs = doc.split('\n')
    sentences = []
    for p in paragraphs:
        sentences.extend(nltk.sent_tokenize(p))

    sentence_list = []
    offset = 0
    for i, s in enumerate(sentences):
        offset = doc.find(s, offset)
        offset_end = offset + len(s)
        assert s == doc[offset: offset_end]
        sentence_list.append((i, s, (offset, offset_end)))
        offset = offset_end
    return sentence_list
