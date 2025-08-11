import hashlib
import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


def generate_permutation_from_key_xof(size, key: bytes, salt: bytes = b""):
    shake = hashlib.shake_256()
    shake.update(key + salt)

    # use 4 bytes per index to generate a sortable list of random ints
    # Each index is mapped to a 32-bit (4-byte) integer — so ints[i] is in the range 0 to 2^32
    # That gives 4.2 billion unique values to sort on IE vocab size = 50k its determincistic and truly random
    derived = shake.digest(size * 4)
    ints = [int.from_bytes(derived[i * 4 : (i + 1) * 4], "big") for i in range(size)]

    perm = sorted(range(size), key=lambda i: ints[i])
    return perm


def permute_bf16_inplace(
    layer: str, num_embeddings: int, hidden_size: int, vocab_size: int, key: bytes
):
    file = layer + ".bin"
    data = np.fromfile(file, dtype=np.uint16)
    data = data.reshape(num_embeddings, hidden_size)

    perm = generate_permutation_from_key_xof(vocab_size, key)
    data[:vocab_size] = data[perm]

    data.tofile(file)


def decrypt_prompt(
    encrypted_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
    key: bytes,
    salt: bytes = b"",
):
    """
    Decrypt a prompt by reversing the token ID permutation applied by `encrypt_prompt`.

    Parameters
    ----------
    prompt : list[int]
        The encrypted prompt as a list of ints.
    tokenizer : transformers.PreTrainedTokenizerBase
        The tokenizer used for both encryption and decryption.
    key, salt : bytes
        Same key/salt used during encryption.

    Returns
    -------
    str
        The decrypted prompt as a decoded string.
    """

    encrypted_ids = np.array(encrypted_ids, dtype=np.int64)

    vocab_size = tokenizer.vocab_size
    perm = generate_permutation_from_key_xof(vocab_size, key, salt)
    perm = np.array(perm, dtype=np.int64)  # forward permutation

    special_ids = set(tokenizer.all_special_ids)
    special_ids_arr = np.array(list(special_ids), dtype=np.int64)

    # mask: True for regular tokens in range [0, vocab_size‑1] and not special
    mask = (encrypted_ids < vocab_size) & ~np.isin(encrypted_ids, special_ids_arr)

    decrypted = np.copy(encrypted_ids)
    decrypted[mask] = perm[encrypted_ids[mask]]

    decrypted_string = tokenizer.decode(decrypted[0], skip_special_tokens=True)

    return decrypted_string


def encrypt_prompt(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
    key: bytes,
    salt: bytes = b"",
) -> Tuple[List[int], str]:
    """
    Encrypt a prompt by permuting its token IDs using NumPy.

    NOTE: This permutes only the vocab space not special tokens for simplicity.
    For extra security tests show you can permute the specal vocab space but
    the LLM must know what EOS is BOS otherwise it gets very confused and has degraded preformance.
    Eather way the info must be leaked to the LLM (provider) so it understands when to stop and start sequences.
    If you add your own specal tokens you may need to adjust this function as well as the lm_head / embedding space to account for the
    new tokens

    Perhaps slot new token in first after normal vocab then just increase embedding size by one (untested)

    tokenizer.add_special_tokens(
        {"my_crazy_new_token": "my_crazy_new_token"}
    )

    We permute all vocab plus the new specal token
    num_embeddings = num_embeddings + 1 = [...normal_vocab, my_crazy_new_token, EOS, BOS]
    then permute_bf16_inplace()

    Returns
    -------
    (List[int], str)

    """

    input_ids = np.array(input_ids, dtype=np.int64)

    vocab_size = tokenizer.vocab_size  # excludes added tokens / specal tokens
    perm = generate_permutation_from_key_xof(vocab_size, key, salt)

    inv_perm = np.empty(vocab_size, dtype=np.int64)
    for i, p in enumerate(perm):
        inv_perm[p] = i

    special_ids = set(tokenizer.all_special_ids)
    special_ids_arr = np.array(list(special_ids), dtype=np.int64)

    mask = (input_ids < vocab_size) & ~np.isin(input_ids, special_ids_arr)

    encrypted_ids = np.copy(input_ids)
    encrypted_ids[mask] = inv_perm[input_ids[mask]]

    encrypted_string = tokenizer.decode(encrypted_ids[0], skip_special_tokens=True)

    return encrypted_ids.tolist(), encrypted_string
