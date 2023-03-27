import subprocess
import sys
import tempfile

from lark import Lark, Tree

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

GRAMMAR = """
start: (triple_line | NEWLINE)+

VERB_LEMMA_LETTERS: LETTER | "+"
participant_literal: participant_list | participant | OMMITTED
participant_list: "[" list_tail  "]"
list_tail: participant ("," list_tail)?
participant: [UCASE_LETTER|"_"]
OMMITTED: "_"
VERB_LEMMA: VERB_LEMMA_LETTERS+
triple_line: triple | triple ","
triple: "(" participant_literal "," (VERB_LEMMA | OMMITTED | option_list) "," participant_literal ")"
option_list: "[" (VERB_LEMMA ","?)+ "]"


// imports WORD from library
%import common.UCASE_LETTER
%import common.LETTER
%import common.NEWLINE

// Disregard spaces in text
%ignore " "
"""


def arg_from_tree(tree):
    if isinstance(tree, Tree):
        return [rule.children[0].value for rule in tree.find_data("participant")]
    else:
        return []


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
        yield arg_from_tree(arg_a), verb_or_choice, arg_from_tree(arg_b)


lark_parser = Lark(GRAMMAR)
# print(list(parsed))


def read_editor_input(prefilled: str):
    text_file = tempfile.NamedTemporaryFile("wt")
    text_file = tempfile.NamedTemporaryFile("wt")
    temp_dir_manager = tempfile.TemporaryDirectory()
    text_file.write(prefilled)
    text_file.flush()
    with temp_dir_manager as temp_dir:
        while True:
            args = [
                "vim",
                "-c",
                f"set undodir={temp_dir}",
                "-c",
                "set undofile",
                text_file.name,
            ]
            ret = subprocess.run(
                args, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr
            )
            text = "".join(open(text_file.name).readlines())
            yield list(get_triples(text)), text


if __name__ == "__main__":
    parsed = get_triples(
        """
        ([A, B], looks+at, _)
        (B, punches, A)
        (A, [hates, loves], C)
    """
    )
    print(list(parsed))
