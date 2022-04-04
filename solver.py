import copy
import csv
import json
from itertools import chain


def format_xp_int(xp_int, sign=True):
    sign_character = '-'
    if xp_int >= 0:
        sign_character = '+' if sign else ''
    xp_int_str = str(abs(xp_int))
    return sign_character + (xp_int_str[:-1] if len(xp_int_str[:-1]) > 0 else '0') + "." + xp_int_str[-1:]


def format_xp_tuple(xp_tuple):
    return '{} xp ({} bonus xp),'.format(format_xp_int(xp_tuple[0]), format_xp_int(xp_tuple[1], False)).ljust(28)


def get_data_points(filename):
    data_points = []
    with open(filename) as csvfile:
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

            data_points.append({'boost_vals': boost_vals, 'xp_vals': xp_vals})
    return data_points


# true if valid model for point, false otherwise
def test_data_point(model, point):
    point_xp_vals = point['xp_vals']
    point_boost_vals = point['boost_vals']
    row_xp = get_xp(point_xp_vals['base'], model, point_boost_vals)
    return point_xp_vals['xp'] == row_xp[0] and (point_xp_vals['(bonus)'] == 0 or point_xp_vals['(bonus)'] == row_xp[1])


# return all
def get_contradictory_test_points(model, points):
    contradictory_data_points = []
    for point in points:
        if not test_data_point(model, point):
            point_xp_vals = point['xp_vals']
            point_boost_vals = point['boost_vals']
            point_xp = get_xp(point_xp_vals['base'], model, point_boost_vals)
            point_xp_vals['calculated_xp'] = point_xp
            point['xp_vals']['discrepancy'] = (point['xp_vals']['xp'] - point_xp[0], point['xp_vals']['(bonus)'] - point_xp[1])
            contradictory_data_points.append({'boost_vals': point_boost_vals, 'xp_vals': point_xp_vals})
    return contradictory_data_points


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
    additive = apply(base, model['additive'], boost_amts)
    multiplicative = apply(base, model['multiplicative'], boost_amts)
    bonus_xp = multiplicative * apply(boost_amts['bxp'], model['bxp'], boost_amts) // 1000
    total = additive + multiplicative + bonus_xp - base
    return total, total - base


def get_xp_old(base, model, boost_amts):
    return apply(base, model['additive'], boost_amts) + apply(base, model['multiplicative'], boost_amts) - base


def get_single_generation_of_successors(model, field, groups):
    for key in model.keys():
        boost_group = get_boost_group(model[key])
        boost_list = boost_group['boosts']
        # group type 2 is default, 1 needs to be specified for the time being
        for trunc_type in range(1 if groups else 2, 3):
            for index in range(len(boost_list) + 1):
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


# pass a string or list of strings to fields to generate all possible models based on those fields
# add dry mode for counting enumerations
def get_successors(starting_model, fields, groups=True):
    if type(fields) is str:
        fields = [fields]
    if len(fields) == 1:
        groups = False
    models = [starting_model]
    for field in fields:  # number of generations of successors
        next_successors = []
        for model in models:  # generate the next generation of successors for all models of the previous generation
            next_successors.append(get_single_generation_of_successors(model, field, groups))
        models = chain.from_iterable(next_successors)  # flatten
    return models


# test every model for every data point, and reject a model when it fails
def filter_models(models, data_points, allowed_failures=0):
    for model in models:
        model_valid = True
        failures = 0
        for point in data_points:
            if not test_data_point(model, point):
                failures += 1
                if failures > allowed_failures:
                    model_valid = False
                    break
        if model_valid:
            yield model


# base xp amounts that might prove useful
activities = {
    "tree": 250,
    "oak": 375,
    "willow": 675,
    "teak": 850,
    "acadia": 920,
    "maple": 1000,
    "mahogany": 1250,
    "eucalyptus": 1650,
    "yew": 1750,
    "bamboo": 2025,
    "magic": 2500,
    "elder": 3250,
    "ivy": 3325,
    "idol": 3835,
    "crystal": 4345,
    "gbamboo": 6555,
}

# ratio boost represented as the numerator over a denominator of 1000
test_boost_amts = dict(  # range,[step]
    wise=40,  # 40
    torstol=5,  # 5-20,5
    outfit=10,  # 10-60,10
    avatar=50,  # 30-60,10
    yak_track=200,  # 100-200,100
    wisdom=25,  # 25
    bxp=0,  # 1000
    premier=0,  # 100
    scabaras=0,  # 100
    prime=0,  # 100-1000,900
    pulse=0,  # 20-100,20
    cinder=0,  # 20-100,20
    worn_cinder=0,  # 1500
    vos=0,  # 200
    coin=0,  # 10-20,10
    sceptre=0,  # 20-40,20
)

sota_model = dict(
    base=[['vos']],
    additive=[['yak_track', 'scabaras', 'prime']],
    multiplicative=[['wise', 'premier', 'torstol', 'outfit', 'pulse'], ['wisdom'], ['avatar']],
    bxp=[['worn_cinder'], ['cinder']]
)

test_model = dict(
    base=[['vos']],
    additive=[['yak_track', 'scabaras', 'prime']],
    multiplicative=[{"boosts": ['wise', 'premier', 'outfit', 'pulse'], "type":1}, ['avatar']],
    bxp=[['worn_cinder'], ['cinder']]
)

counting_model = dict(first=[])

# pasted boosted amounts
test_boost_amts = {'bxp': 0, 'dxp': 0, 'bomb': 0, 'yak_track': 200, 'torstol': 5, 'premier': 0, 'avatar': 50, 'worn_pulse': 0, 'pulse': 0, 'worn_cinder': 0, 'cinder': 0, 'sceptre': 0, 'coin': 0, 'outfit': 10, 'raf': 0, 'aura': 0, 'wise': 40, 'shared': 0, 'vos': 0, 'brawlers': 0, 'inspire': 0, 'wisdom': 25, 'scabaras': 0, 'prime': 0, 'temp': 0}

# base xp for singular data point tests
test_base_xp = activities["idol"]
# None to show all successor models
expected = None


# number of fields greatly increases search space, >O(n^n)
# 5 fields crashed my computer
fields_to_add = ['wisdom', 'torstol']
# whether or not to generate "sum without round" additive groups within multiplicative groups
groups = True
allowed_failures = 12

print('Starting model:')
print(json.dumps(test_model))

print('Loading data to test successor models against:')
data_points = get_data_points('data.csv')
# don't want to look at coin yet
data_points = list(filter(lambda data_point: data_point['boost_vals']['temp'] == 0, data_points))

print('Generating and testing successor models by adding {}...'.format(fields_to_add))
all_successors = list(get_successors(test_model, fields_to_add, True))
successful_models = list(filter_models(all_successors, data_points, allowed_failures))
print("{} successful models".format(len(successful_models)))
[print(model) for model in successful_models]

error_points = get_contradictory_test_points(sota_model, data_points)
# remove the coin/sceptre data for now
error_points = list(filter(lambda error_point: error_point['boost_vals']['temp'] == 0, error_points))
print('{} contradictory samples calculated with sota model, printing...'.format(len(error_points)))
for error_point_dict in error_points:
    xp_vals = error_point_dict['xp_vals']
    discrepancy = format_xp_tuple(xp_vals['discrepancy'])
    calculated = format_xp_tuple(xp_vals['calculated_xp'])
    observed = format_xp_tuple((xp_vals['xp'], xp_vals['(bonus)']))
    print('discrepancy: {} calculated: {} observed: {} base: {}, boosts:{}'
          .format(discrepancy, calculated, observed, xp_vals['base'], error_point_dict['boost_vals']))



# code that tests a specific data point
print('Showing successor model results for specific data point')
xp = get_xp(test_base_xp, test_model, test_boost_amts)
print('Test boosts:')
print(json.dumps(test_boost_amts, indent=2))
print('Starting model test result:', xp)

test_point_successors = [(get_xp(test_base_xp, model, test_boost_amts), model) for model in all_successors]
matching_successors = list(
    filter(lambda entry: expected is not None and entry[0][0] == expected, test_point_successors))
if len(matching_successors) == 0:
    to_print = test_point_successors
    print('No successor models which add {}, matching {} boosted xp for {} base xp, found.'
          .format(fields_to_add, expected, test_base_xp))
    print('Printing all {} successor models and their results:'.format(len(to_print)))
else:
    to_print = matching_successors
    print('{} models which add {}, matching {} boosted xp for {} base xp, found:'
          .format(len(to_print), fields_to_add, expected, test_base_xp))
to_print = test_point_successors if len(matching_successors) == 0 else matching_successors
[print(format_xp_tuple(entry[0]) + str(entry[1])) for entry in to_print]
print()
