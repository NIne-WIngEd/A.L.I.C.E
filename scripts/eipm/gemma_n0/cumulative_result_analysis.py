"""Compare complete matched saved objectives without inventing a pass threshold.

Native losses already apply certainty weighting within a case. These are
arithmetic means over matched case losses, not pooled fact losses or accuracy.
Different loss families stay separate. No targets or model state are changed.
"""
import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _index(rows):
    _require(type(rows) is list and bool(rows), 'complete nonempty objective rows required')
    result = {}
    batches = set()
    for row in rows:
        _require(type(row) is dict and all(key in row for key in (
            'packet', 'phase', 'case_id', 'case_sha256', 'batch_sha256', 'total_loss',
            'field_loss', 'available_counts', 'descriptive_training_targets_not_independent_qualification')),
            'objective row is incomplete')
        _require(all(type(row[key]) is str and row[key] for key in
            ('packet', 'phase', 'case_id', 'case_sha256', 'batch_sha256')), 'objective identity is incomplete')
        key = (row['packet'], row['phase'], row['case_id'])
        _require(key not in result and row['batch_sha256'] not in batches, 'duplicate objective case or batch')
        _require(row['descriptive_training_targets_not_independent_qualification'] is True,
                 'saved teacher fit is not independent qualification')
        _require(_number(row['total_loss']) and type(row['field_loss']) is dict
                 and type(row['available_counts']) is dict
                 and bool(row['field_loss']) and set(row['field_loss']) == set(row['available_counts']),
                 'objective loss/coverage is incomplete or nonfinite')
        _require(all(type(field) is str and field and _number(value)
                     for field, value in row['field_loss'].items()), 'field loss is invalid')
        _require(all(type(count) is int and count >= 0 for count in row['available_counts'].values()),
                 'objective coverage is invalid')
        result[key] = row
        batches.add(row['batch_sha256'])
    return result


def _measure(pairs):
    count = len(pairs)
    return {'case_count': count,
        'before_mean_case_loss': math.fsum(a for a, _ in pairs) / count,
        'after_mean_case_loss': math.fsum(b for _, b in pairs) / count,
        'mean_case_loss_change': math.fsum(b - a for a, b in pairs) / count,
        'improved_cases': sum(b < a for a, b in pairs),
        'worsened_cases': sum(b > a for a, b in pairs),
        'unchanged_cases': sum(b == a for a, b in pairs)}


def compare_objectives(before, after):
    left, right = _index(before), _index(after)
    _require(left.keys() == right.keys(), 'before/after objective coverage changed')
    groups = {}
    for key in sorted(left):
        first, second = left[key], right[key]
        _require(all(first[name] == second[name] for name in
                     ('case_sha256', 'batch_sha256', 'available_counts'))
                 and first['field_loss'].keys() == second['field_loss'].keys(),
                 'before/after objective identity or reviewed coverage changed')
        group = groups.setdefault('/'.join(key[:2]), {'total': [], 'fields': {}, 'counts': {}})
        group['total'].append((first['total_loss'], second['total_loss']))
        for field, value in first['field_loss'].items():
            group['fields'].setdefault(field, []).append((value, second['field_loss'][field]))
            group['counts'][field] = group['counts'].get(field, 0) + first['available_counts'][field]
    return {'case_count': len(left), 'independent_qualification': False,
        'aggregation': 'arithmetic_mean_of_native_certainty_weighted_case_losses_not_pooled_fact_loss',
        'groups': {name: {'total': _measure(group['total']),
            'fields': {field: {**_measure(pairs), 'available_facts': group['counts'][field]}
                       for field, pairs in sorted(group['fields'].items())}}
                   for name, group in sorted(groups.items())}}
