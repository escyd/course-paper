"""Pure observation of quoted liquidity; durations use recorded step samples."""
def quote_state(market):
    bid_empty = not bool(market.order_book['bid'])
    ask_empty = not bool(market.order_book['ask'])
    return dict(bid_empty=bid_empty, ask_empty=ask_empty,
                two_sided=not (bid_empty or ask_empty),
                empty_both=bid_empty and ask_empty,
                state=('empty_both' if bid_empty and ask_empty else
                       'one_sided' if bid_empty or ask_empty else 'two_sided'))


def empty_spells(flags):
    """Zero-based start, sample duration, and right censoring at observation end."""
    result, start = [], None
    for step, empty in enumerate([*flags, False]):
        if empty and start is None:
            start = step
        elif not empty and start is not None:
            result.append(dict(start=start, duration=step-start,
                               right_censored=step == len(flags)))
            start = None
    return result


def liquidity_summary(flags):
    spells = empty_spells(flags)
    return dict(empty_steps=sum(bool(x) for x in flags), empty_periods=len(spells),
                longest_empty_period=max((x['duration'] for x in spells), default=0),
                empty_fraction=sum(bool(x) for x in flags)/len(flags) if flags else 0.)


def quote_history_summary(bid_empty, ask_empty):
    """Mutually exclusive sampled states; no inference about intrastep spells."""
    if len(bid_empty) != len(ask_empty):
        raise ValueError('Liquidity histories must have equal lengths')
    missing = [bool(b or a) for b, a in zip(bid_empty, ask_empty)]
    both = sum(bool(b and a) for b, a in zip(bid_empty, ask_empty))
    return dict(liquidity_summary(missing), two_sided_steps=len(missing)-sum(missing),
                one_sided_steps=sum(missing)-both, empty_both_steps=both,
                fallback_observation_fraction=sum(missing)/len(missing) if missing else 0.,
                two_sided_recoveries=sum(x and not y for x, y in zip(missing, missing[1:])))
