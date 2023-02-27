from lark import Lark

GRAMMAR = """
start: (triple_line | NEWLINE)+

VERB_LEMMA_LETTERS: LETTER | "+"
participant: [UCASE_LETTER|"_"]
VERB_OMMITTED: "_"
VERB_LEMMA: VERB_LEMMA_LETTERS+
triple_line: triple | triple ","
triple: "(" participant "," (VERB_LEMMA | VERB_OMMITTED| option_list) "," participant ")"
option_list: "[" (VERB_LEMMA ","?)+ "]"


// imports WORD from library
%import common.UCASE_LETTER
%import common.LETTER
%import common.NEWLINE

// Disregard spaces in text
%ignore " "
"""


def get_triples(text):
    parsed = lark_parser.parse(text)
    for triple in parsed.find_data("triple"):
        arg_a, verb_or_choice, arg_b = triple.children
        assert len(arg_a.children) == 1
        arg_a = arg_a.children[0]
        assert len(arg_b.children) == 1
        arg_b = arg_b.children[0]
        try:
            verb_or_choice = verb_or_choice.value
        except AttributeError:
            verb_or_choice = [v.value for v in verb_or_choice.children]
        yield arg_a.value, verb_or_choice, arg_b.value


lark_parser = Lark(GRAMMAR)
parsed = get_triples(
    """
    (A, looks+at, B)
    (B, punches, A)
    (A, [hates, loves], C)
"""
)

print(list(parsed))
