"""Build a standalone viewer from the exact generated launch manifest."""
import json
from pathlib import Path
from kernel_codegen import function_source


EXPLANATIONS = {
    'embedding': dict(title='Embedding lookup', forward='X[t, d] = E[token[t], d]', backward='dE[token[t], d] += dX[t, d]',
        saved='Integer token IDs.', why='Gather becomes scatter-add. Repeated tokens contribute to the same row. E is also the language-model head: add this contribution to its existing gradient. Token IDs have no gradient.'),
    'linear': dict(title='Linear projection', forward='Y = X Wᵀ', backward='dX += dY W     ·     dW += dYᵀ X',
        saved='X and W. The output Y is not needed by this local backward.', why='One forward matrix multiplication becomes two backward matrix multiplications. dX passes learning signals to the previous operation; dW tells the optimizer how to change this layer. dW sums over every batch/token row.'),
    'rmsnorm': dict(title='RMS normalization', forward='r = 1 / √(mean(x²) + ε)\ny = w ⊙ x · r',
        backward='dx += r · w ⊙ dy − x · r³ · mean(dy ⊙ w ⊙ x)\ndw += Σrows dy ⊙ x · r',
        saved='Input x, scale w, and one reciprocal RMS r per token.', why='Changing one input also changes the normalization of every feature in that token. The second term accounts for that coupling. Only a reduction is needed; we never construct the full Jacobian.'),
    'rope': dict(title='Rotary position embedding', forward='y = R(θ) x', backward='dx += R(θ)ᵀ dy = R(−θ) dy',
        saved='Position index and fixed frequencies; sin/cos are recomputed.', why='A rotation preserves lengths. Its backward rotates the incoming gradient in the opposite direction. Llama-3 frequency scaling and the same positions apply to both passes.'),
    'scores': dict(title='Causal QK scores + GQA', forward='S[h] = Q[h] K[group(h)]ᵀ / √D\nS[t, s] = −∞ for s > t',
        backward='dQ[h] += dS[h] K[group(h)] / √D\ndK[g] += Σh in group(g) dS[h]ᵀ Q[h] / √D',
        saved='Rotated Q and K; the causal mask is reconstructed from indices.', why='Several query heads read the same K head. Their gradients must be summed into that head. Masked scores are constants and have zero gradient.'),
    'softmax': dict(title='Causal softmax', forward='P = exp(S − max(S)) / Σ exp(S − max(S))',
        backward='dS += P ⊙ (dP − Σkeys(dP ⊙ P))',
        saved='Probabilities P.', why='Softmax couples the keys in each row. Subtract the probability-weighted gradient average, then multiply by P. The row sum of dS is approximately zero; masked positions stay exactly zero.'),
    'values': dict(title='Mix attention values', forward='O[h] = P[h] V[group(h)]   (causal keys only)',
        backward='dP[h] += dO[h] V[group(h)]ᵀ\ndV[g] += Σh in group(g) P[h]ᵀ dO[h]',
        saved='P and V.', why='One path tells attention how to change its probabilities; the other tells it how to change the values. Shared V heads receive a sum across query heads. This implementation skips causal-mask entries in both passes.'),
    'add': dict(title='Residual connection', forward='y = a + b', backward='da += dy     ·     db += dy',
        saved='No forward values.', why='A merge in forward becomes a fork in backward. Send the same incoming gradient down both paths. Later, the skip-path and transformed-path contributions meet and add together; do not overwrite either one.'),
    'swiglu': dict(title='SwiGLU', forward='s = sigmoid(gate)\ny = gate ⊙ s ⊙ up',
        backward='dgate += dy ⊙ up ⊙ (s + gate ⊙ s ⊙ (1−s))\ndup += dy ⊙ gate ⊙ s',
        saved='gate and up; sigmoid is recomputed.', why='Apply the product rule to the two branches, then differentiate SiLU(gate). Both the gate and up projection weights need gradients.'),
    'cross_entropy': dict(title='Next-token cross entropy', forward='loss[t] = logsumexp(logits[t]) − logits[t, target[t]]',
        backward='dlogits[t] += dloss[t] · (softmax(logits[t]) − one_hot(target[t]))',
        saved='Logits and shifted integer targets.', why='The correct class receives p−1; every other class receives p. Stable row reductions avoid explicitly storing a vocabulary-wide Jacobian. All token rows are valid; this demo has no padding/ignore labels.'),
    'mean': dict(title='Mean loss / start of backward', forward='L = Σt loss[t] / (batch · sequence)', backward='Seed dL = 1\ndloss[t] += dL / (batch · sequence)',
        saved='The number of predicted tokens.', why='This is where backward starts. dL/dL is 1. The mean divides that signal among the tokens exactly once; the rest of the graph propagates it by the chain rule.'),
}


def extract_kernel(source, name):
    return function_source(source, name)


def write_viewer(manifest, directory, kernels):
    functions = {e['kernel']: extract_kernel(kernels, e['kernel'])
                 for r in manifest['operations'] for phase in ('forward', 'backward') for e in r[phase]}
    functions.update({name: extract_kernel(kernels, name) for name in ('block_sum', 'block_max', 'rope_frequency')})
    data = json.dumps(dict(manifest=manifest, explanations=EXPLANATIONS, kernels=functions)).replace('</', '<\\/')
    template = (Path(__file__).parent / 'docs/viewer.html').read_text()
    (Path(directory) / 'index.html').write_text(template.replace('/*__DATA__*/', data))
