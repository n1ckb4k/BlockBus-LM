# Data

`data/tinyshakespeare.txt` is the Tiny Shakespeare character corpus distributed with
Andrej Karpathy's `char-rnn` examples:

- Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Size: 1,115,394 bytes
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- Content: public-domain William Shakespeare plays/poems, concatenated.

The repository includes the file so all experiments are reproducible offline. If it is
missing, run `python scripts/download_data.py`. The downloader verifies the SHA-256.

Splits are contiguous and fixed:

- train: first 90% (1,003,854 characters)
- validation: next 5% (55,769 characters)
- test: final 5% (55,771 characters)

The character vocabulary is built from the training split only (65 characters plus an
`<unk>` token, so 66 model tokens). No characters in validation/test were unknown.
