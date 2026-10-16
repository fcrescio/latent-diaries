"""Offline interventions on an existing continuous prior; never training code."""
import torch


@torch.no_grad()
def trace_prior(model, semantic, gold_prefix, seed, *, gold_history=False, reset_position=None):
    if semantic.shape != (1, 768) or gold_prefix.shape[0] != 1:
        raise ValueError('Batch-one diagnostic inputs required')
    gold = model.encode(gold_prefix)[0]
    previous = gold.new_zeros(1, model.dimension)
    recurrent, vectors, contexts, records = None, [], [], []
    generator = torch.Generator(device=gold.device).manual_seed(seed)
    for position in range(len(gold) + 1):
        # Replacing vector j changes the input at j+1; never resets recurrent memory.
        if position and (gold_history or position - 1 == reset_position):
            previous = gold[position - 1:position]
        inputs = (model.previous(previous) + model.features(semantic) + model.position.weight[position])[:, None]
        context, recurrent = model.gru(inputs, recurrent)
        context = context[:, 0];contexts.append(context[0])
        means, routing = model.predictions(context)
        logits, indices = routing[0].topk(min(4, model.modes))
        selected = int(torch.multinomial((logits / 0.5).softmax(-1), 1, generator=generator))
        mode = int(indices[selected]);prediction = means[0, mode]
        probabilities = routing[0].softmax(-1)
        record = {'position': position, 'selected_mode': mode,
                  'routing_top1': int(routing[0].argmax()),
                  'routing_entropy': float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()),
                  'stop_probability': float(model.stop(context).sigmoid()[0, 0]),
                  'candidate_variance': float(means[0].var(0, unbiased=False).mean())}
        if position < len(gold):
            errors = ((means[0] - gold[position]) * model.latent_scale).square().sum(-1) / model.mean.numel()
            record.update(selected_normalized_mse=float(errors[mode]), oracle_best_normalized_mse=float(errors.min()),
                          oracle_best_mode=int(errors.argmin()), routing_matches_best=mode == int(errors.argmin()))
            vectors.append(prediction);previous = prediction[None]
        records.append(record)
    latent = torch.stack(vectors)[None]
    return {'records': records, 'prefix': model.decode(latent), 'contexts': torch.stack(contexts),
            'forced_length': len(gold), 'stop_ignored_for_aligned_diagnostic': True}


def compare_traces(reference, trace):
    result = []
    cumulative = 0.0
    for i, row in enumerate(trace['records']):
        if 'selected_normalized_mse' in row:
            cumulative += row['selected_normalized_mse']
        result.append({**row, 'cumulative_normalized_mse': cumulative,
                       'hidden_rms_from_gold_history': float((trace['contexts'][i] - reference['contexts'][i]).square().mean().sqrt()),
                       'mode_agrees_gold_history': row['selected_mode'] == reference['records'][i]['selected_mode']})
    return result
