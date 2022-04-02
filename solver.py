import copy
import csv
from itertools import chain


def format_xp_int(xp_int, sign=True):
    sign_character = '-'
    if xp_int >= 0:
        sign_character = '+' if sign else ''
    xp_int_str = str(abs(xp_int))
    return sign_character + (xp_int_str[:-1] if len(xp_int_str[:-1]) > 0 else '0') + "." + xp_int_str[-1:]


def format_xp_tuple(xp_tuple):
    return '{} xp ({} bonus xp),'.format(format_xp_int(xp_tuple[0]), format_xp_int(xp_tuple[1], False)).ljust(28)


def get_contradictory_test_points(filename):
    contradictory_test_points = []
    with open('data.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            xp_vals = {}
            boost_vals = {}
            for key in row:
                if key in ['xp', '(bonus)', 'base']:
                    if row[key] is None or row[key] == '':
                        xp_vals[key] = 0
                    else:
                        xp_vals[key] = round(float(row[key]) * 10)
                else:
                    if row[key] is None or row[key] == '':
                        boost_vals[key] = 0
                    else:
                        boost_vals[key] = round(float(row[key]) * 1000)
            row_xp = get_xp(xp_vals['base'], correct_model, boost_vals)
            xp_vals['calculated_xp'] = row_xp
            if xp_vals['xp'] != row_xp[0] or (xp_vals['(bonus)'] != 0 and xp_vals['(bonus)'] != row_xp[1]):
                observed_bonus = xp_vals['(bonus)'] if xp_vals['(bonus)'] is not None else 0
                xp_vals['discrepancy'] = (xp_vals['xp'] - row_xp[0], observed_bonus - row_xp[1])
                contradictory_test_points.append({'boost_vals': boost_vals, 'xp_vals': xp_vals})
    return contradictory_test_points


# ratio boost represented as the numerator over a denominator of 1000
test_boost_amts = dict( #range,[step]
    wise = 40, #40
    torstol = 5, #5-20,5
    outfit = 10, #10-60,10
    avatar = 50, #30-60,10
    yak_track = 200, #100-200,100
    wisdom = 25, #25
    bxp = 0, #1000
    premier = 0, #100
    scabaras = 0, #100
    prime = 0, #100-1000,900
    pulse = 0, #20-100,20
    cinder = 0, #20-100,20
    worn_cinder = 0, #1500
    vos = 0, #200
    coin = 0, #10-20,10
    sceptre = 0, #20-40,20
)


# apply function where every nested boost is multiplied by the base amount, then truncated, then added together
def apply2(base, boosts, boost_amts):
    for elt in boosts:
        base = base + sum(base * boost_amts[i] // 1000 for i in elt)
    return base


# apply function where every nested boost is added together, then added to 1000, and then truncated
def apply1(base, boosts, boost_amts):
    for elt in boosts:
        base = (base * (1000 + sum(boost_amts[i] for i in elt))) // 1000
    return base


# convert lists to dict format for internal use
def get_boost_group(list_or_dict):
    if type(list_or_dict) is list:
        inner_boost_list = list_or_dict
        trunc_type = 2
    elif type(list_or_dict) is dict:
        inner_boost_list = list_or_dict['boosts']
        trunc_type = list_or_dict['type']
    else:
        raise Exception('malformed model:', list_or_dict)
    return {"boosts": inner_boost_list, "type": trunc_type}


def apply(base, boosts, boost_amts):
    for elt in boosts:
        boost_group = get_boost_group(elt)
        if boost_group['type'] == 1:
            # apply function where every nested boost is added together, then added to 1000, and then truncated
            base = (base * (1000 + sum(boost_amts[i] for i in boost_group['boosts']))) // 1000
        else:
            # apply function where every nested boost is multiplied by the base amount, then truncated, then summed
            base = base + sum(base * boost_amts[i] // 1000 for i in boost_group['boosts'])
    return base


def get_xp(base, model, boost_amts):
    base = apply(base, model['base'], boost_amts)
    total = apply(base, model['additive'], boost_amts) \
        + apply(base, model['multiplicative'], boost_amts) \
        * (1000 + apply(boost_amts['bxp'], model['bxp'], boost_amts))//1000 \
        - base
    return total, total - base


def get_xp_old(base, model, boost_amts):
    return apply(base, model['additive'], boost_amts) + apply(base, model['multiplicative'], boost_amts) - base


def get_successors(model, field, groups=False):
    for key in model.keys():
        boost_group = get_boost_group(model[key])
        boost_list = boost_group['boosts']
        for trunc_type in range(1 if groups else 2, 3):
            for index in range(len(boost_list)+1):
                c = copy.deepcopy(model)
                if trunc_type == 1:
                    c[key].insert(index, {'boosts': [field], 'type': 1})
                else:
                    c[key].insert(index, [field])
                yield c
        for index in range(len(model[key])):
            c = copy.deepcopy(model)
            if type(c[key][index]) is list:
                c[key][index].append(field)
            else:
                c[key][index]['boosts'].append(field)
            yield c


correct_model = dict(
    base = [['vos']],
    additive = [['yak_track', 'scabaras', 'prime']],
    multiplicative = [['wise', 'premier', 'torstol', 'outfit', 'pulse', 'temp'], ['wisdom'], ['avatar']],
    bxp = [['worn_cinder'], ['cinder']]
)

correct_model = dict(
    base = [['vos']],
    additive = [['yak_track', 'scabaras', 'prime']],
    multiplicative = [['wise', 'premier', 'torstol', 'outfit', 'pulse', 'temp'], ['avatar']],
    bxp = [['worn_cinder'], ['cinder']]
)

# correct_model = {'base': [['vos']], 'additive': [['yak_track', 'scabaras', 'prime']], 'multiplicative': [['wise', 'premier', 'torstol', 'outfit', 'pulse'], ['wisdom'], ['coin', 'sceptre'], ['avatar']], 'bxp': [['worn_cinder'], ['cinder']]}
test_boost_amts = {'bxp': 0, 'dxp': 0, 'bomb': 0, 'yak_track': 200, 'torstol': 0, 'premier': 100, 'avatar': 50, 'worn_pulse': 0, 'pulse': 100, 'worn_cinder': 0, 'cinder': 100, 'sceptre': 0, 'temp': 30, 'outfit': 50, 'raf': 0, 'aura': 0, 'wise': 40, 'shared': 0, 'vos': 0, 'brawlers': 0, 'inspire': 0, 'wisdom': 0, 'scabaras': 100, 'prime': 0}

base_xp = 1750
expected = 2334

xp = get_xp(base_xp, correct_model, test_boost_amts)
print('original model', (format_xp_tuple(xp)))

# successors = get_successors(correct_model, 'coin')
# successors = chain.from_iterable([get_successors(successor, 'sceptre') for successor in successors])
# successors = [(get_xp(base_xp, model, test_boost_amts), model) for model in successors]
successors = [(get_xp(base_xp, model, test_boost_amts), model) for model in get_successors(correct_model, 'wisdom')]
matching_successors = list(filter(lambda entry: expected is not None and entry[0][0] == expected, successors))
to_print = successors if len(matching_successors) == 0 else matching_successors

print('Successor models:')
[print(format_xp_tuple(entry[0]) + str(entry[1])) for entry in to_print]
print()
error_points = get_contradictory_test_points('data.csv')
print('{} contradictory samples calculated with current best model, printing...'.format(len(error_points)))
for error_point_dict in error_points:
    xp_vals = error_point_dict['xp_vals']
    discrepancy = format_xp_tuple(xp_vals['discrepancy'])
    calculated = format_xp_tuple(xp_vals['calculated_xp'])
    observed = format_xp_tuple((xp_vals['xp'], xp_vals['(bonus)']))
    print('discrepancy: {} calculated: {} observed: {} base: {}, boosts:{}'
          .format(discrepancy, calculated, observed, xp_vals['base'], error_point_dict['boost_vals']))
