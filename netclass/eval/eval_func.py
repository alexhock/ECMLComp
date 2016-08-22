__author__ = 'AlexH'

def _harm_mean(a, b, eps=1e-15):
    return (2*a*b/(a+b+eps))


def _metrics(gold, pred, cat, eps=1e-15):
    TP = 0.
    FP = 0.
    FN = 0.
    for g, p in zip(gold, pred):
        if p == cat:
            if g == p:
                TP += 1
            else:
                FP += 1
        elif g == cat:
            FN += 1
    prec = TP/(TP+FP+eps)
    rec = TP/(TP+FN+eps)
    f1 = _harm_mean(prec, rec)
    return (prec, rec, f1, int(TP+FN), TP, FP, FN)

def get_f1_macro(y_gold, y_pred):
    results = {}
    for cat in sorted(set(y_gold)):
        if cat != 0:
            results[cat] = _metrics(y_gold, y_pred, cat)

    tot_TP = sum(results[cat][4] for cat in results)
    tot_FP = sum(results[cat][5] for cat in results)
    tot_FN = sum(results[cat][6] for cat in results)
    micro_prec = tot_TP/(tot_TP+tot_FP)
    micro_rec = tot_TP/(tot_TP+tot_FN)
    macro_prec = sum(results[cat][0] for cat in results)/len(results)
    macro_rec = sum(results[cat][1] for cat in results)/len(results)
    f1_macro = _harm_mean(macro_prec, macro_rec)
    #f1_macro = int(tot_TP+tot_FN)

    return f1_macro


def eval(y_gold, y_pred):

    if len(y_gold) != len(y_pred):
        print ("prediction is not the same length as gold {0} {1}".format(len(y_pred), len(y_gold)))
        return

    print('Detailed Report:')
    header = '{:^12}||  {:^12}||  {:^12}||  {:^12}||  {:^12}||'
    content = '{:<12}||  {:>12.4f}||  {:>12.4f}||  {:>12.4f}||  {:>12}||'

    print(header.format('category', 'precision',
                        'recall', 'F1', 'examples'))
    print('='*75+'==')
    results = {}
    for cat in sorted(set(y_gold)):
        if cat != 0:
            results[cat] = _metrics(y_gold, y_pred, cat)
            print(content.format(cat, *results[cat]))

    print('='*75+'==')

    tot_TP = sum(results[cat][4] for cat in results)
    tot_FP = sum(results[cat][5] for cat in results)
    tot_FN = sum(results[cat][6] for cat in results)
    micro_prec = tot_TP/(tot_TP+tot_FP)
    micro_rec = tot_TP/(tot_TP+tot_FN)
    macro_prec = sum(results[cat][0] for cat in results)/len(results)
    macro_rec = sum(results[cat][1] for cat in results)/len(results)

    print(content.format('Macro',
                         macro_prec,
                         macro_rec,
                         _harm_mean(macro_prec, macro_rec),
                         int(tot_TP+tot_FN)))
    print(content.format('Micro',
                         micro_prec,
                         micro_rec,
                         _harm_mean(micro_prec, micro_rec),
                         int(tot_TP+tot_FN)))
    print('='*75+'==')

