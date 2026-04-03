import os, json

def safe_pct(num, den):
    return round(100 * num / den, 1) if den > 0 else 0.0

files = os.listdir('results')
output = ["Model\tPrompting Variants\tOverall\tOverallIF\tEasy\tEasyIF\tHard\tHardIF\tShort\tShortIF\tMedium\tMediumIF\tLong\tLongIF"]
compensated = False

for file in files:
    filename = os.path.join('results', file)
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    # instruct failure
    easy_if, hard_if, short_if, medium_if, long_if = 0, 0, 0, 0, 0
    for pred in pred_data:
        acc = int(pred['judge'])
        follow = int(pred['follow_instruction'])
        instr_fail = 1 - follow
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
            easy_if += instr_fail
        else:
            hard += 1
            hard_acc += acc
            hard_if += instr_fail

        if pred['length'] == "short":
            short += 1
            short_acc += acc
            short_if += instr_fail
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
            medium_if += instr_fail
        else:
            long += 1
            long_acc += acc
            long_if += instr_fail

    overall = easy + hard
    overall_acc = easy_acc + hard_acc
    overall_if = easy_if + hard_if

    name = file.split('_', 1)[0]
    prompting_variants = file.split("_", 1)[1].rsplit(".", 1)[0]
    output.append(
        f'{name}\t{prompting_variants}\t'
        f'{safe_pct(overall_acc, overall)}\t{safe_pct(overall_if, overall)}\t'
        f'{safe_pct(easy_acc, easy)}\t{safe_pct(easy_if, easy)}\t'
        f'{safe_pct(hard_acc, hard)}\t{safe_pct(hard_if, hard)}\t'
        f'{safe_pct(short_acc, short)}\t{safe_pct(short_if, short)}\t'
        f'{safe_pct(medium_acc, medium)}\t{safe_pct(medium_if, medium)}\t'
        f'{safe_pct(long_acc, long)}\t{safe_pct(long_if, long)}'
    )

open('result.tsv', 'w', encoding='utf-8').write('\n'.join(output))
