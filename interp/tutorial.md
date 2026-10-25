# teaching order

framing: every component is "what can this write to the residual stream?"

## 0. embedding → unembedding (no layers)

### the residual stream starts as the token
### the unembedding reads the stream
### lookup task: chance, nothing moves between positions

## 1. one head, one layer

### what one head can write: a copy of what's already at some other position
### hidden digit: one head solves it (positive control)
### lookup task: fails, "comes after f" isn't written at the digit's position

## 2. two layers, one head each

### layer 1 writes "my key was f" into the digit's position
### layer 2 searches for it and copies the digit
### why not two heads in one layer? width adds lookups, depth lets a lookup use another's result
### where does "one back" come from? position embedding, and the mask leaks order too
    - remove position: limps, 38–61%
    - remove position and mask: ~37%, no order at all

## 3. adding an MLP

### two-key sum: attention-only fails, it can only copy tokens that already exist
### one head per layer plus an MLP solves it: the MLP writes a token nobody had

## 4. two heads per layer

### two-key difference: order matters, so a 50/50 blend of both values isn't enough
### two layers, one head each: usually grabs only one value, stuck at ~42%
### sometimes it finds a lopsided blend (1/3 + 2/3) the MLP can untangle: 89–97%, never 100%
### two heads in layer 2: each fetches one value, 100%
### a third one-head layer also works: width does lookups in parallel, depth in sequence
