# A Sentence to Sentence Baseline Model

This model is intended to implement a baseline using encoder from [ON_LSTM](https://github.com/yikangshen/Ordered-Neurons) and decoder from [non-monotonic generation](https://github.com/wellecks/nonmonotonic_text).

We choose the penn dataset which is the same as in [ON_LSTM](https://github.com/yikangshen/Ordered-Neurons).

To fit the dataset, we had to do a few modification with the encoder and decoder models.

## Project Structure

```plaintext
temp_model
    |
    +-----data
            +-----penn
                    |------train.txt
                    |------train.jsonl
                    |------valid.txt
                    |------valid.jsonl
                    |------test.txt
                    |------test.jsonl
    |
    |-----archive.txt       # save the unused code snippets
    |-----args.py
    |-----attention.py      # from nonmonotonic-text
    |-----data.py           # from Ordered-Neuron
    |-----locked_dropout.py # from Ordered-Neuron
    |-----losses.py         # from nonmonotonic-text
    |-----main.py           # from Ordered-Neuron, but modified a lot on structure
    |-----model.py          # mixed code from both repos
    |-----note.txt          # some notes while developing this repo
    |-----ON_LSTM.py        # from Ordered-Neuron
    |-----oracle.py         # from nonmonotonic-text
    |-----README.md         # this file
    |-----samplers.py       # from nonmonotonic-text
    |-----splitcross.py     # from Ordered-Neuron
    |-----tree.py           # from nonmonotonic-text
    |-----utils.py          # mixed code from both repos
```
