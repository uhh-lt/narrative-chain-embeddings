from typing import Optional

import torch


class MuseText:
    def __init__(self, path):
        self.embs = {}
        in_file = open(path)
        self.oov_counter = 0
        self.total_counter = 0
        next(in_file)
        for line in in_file:
            word, *vec = line.split(" ")
            vec_tensor = torch.tensor([float(s) for s in vec])
            self.embs[word] = vec_tensor
        self.ones = torch.ones_like(vec_tensor)

    def get_oov_ratio(self):
        return self.oov_counter / self.total_counter

    def get_word_vector(self, word, default: Optional[str] = "ones"):
        self.total_counter += 1
        try:
            return self.embs[word]
        except KeyError:
            try:
                return self.embs[word.replace("#", "")]
            except KeyError as e:
                self.oov_counter += 1
                if default == "ones":
                    return self.ones
                else:
                    raise e
