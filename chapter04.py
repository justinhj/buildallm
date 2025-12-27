import torch


def generate_text_simple(model, idx,
                         max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Crop to the context size
        with torch.no_grad():  # Disable auto grad for prediction
            logits = model(idx_cond)  # Get the logits for the the input

        # Focus ONLY on the last time step
        # (Batch, sequence, tokens) -> we want the last index of Internal_Length

        logits = logits[:, -1, :]

        # Softmax - turn the logits into probabilities
        probas = torch.softmax(logits, dim=-1)
        # Take the most probable new token as the next token...
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # Append it to the input and continue the loop
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
