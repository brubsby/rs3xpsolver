import copy
import csv
import json
from itertools import chain, combinations
import textwrap


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def format_xp_int(xp_int, sign=True):
    sign_character = '-'
    if xp_int >= 0:
        sign_character = '+' if sign else ''
    xp_int_str = str(abs(xp_int))
    return sign_character + (xp_int_str[:-1] if len(xp_int_str[:-1]) > 0 else '0') + "." + xp_int_str[-1:]


def format_xp_tuple(xp_tuple):
    return '{} xp ({} bonus xp),'.format(format_xp_int(xp_tuple[0]), format_xp_int(xp_tuple[1], False)).ljust(28)


def flatten(to_flatten):
    for x in to_flatten:
        if hasattr(x, '__iter__') and not isinstance(x, str) and not isinstance(x, tuple):
            for y in flatten(x):
                yield y
        else:
            yield x


# order a list of tuples such that the transition between elements is minimized
# prioritizing elements in the order of field_order
# helpful for ordering a list of generated experiments such that annoying to swap boosts are toggled infrequently
def minimize_transitions(tuple_list, field_order):
    tuple_lists = [tuple_list]
    # whether the field is included last (true), or not included last (false), in the previous list
    for field in field_order:
        next_tuple_lists = []
        ending_with = False
        for tuple_list in tuple_lists:
            if len(tuple_list) == 1:
                ending_with = field in tuple_list[0]
                next_tuple_lists.append(tuple_list)
            elif len(tuple_list) == 0:
                continue
            else:
                if ending_with:
                    next_tuple_lists.append(list(filter(lambda this_tuple: field in this_tuple, tuple_list)))
                    next_tuple_lists.append(list(filter(lambda this_tuple: field not in this_tuple, tuple_list)))
                else:
                    next_tuple_lists.append(list(filter(lambda this_tuple: field not in this_tuple, tuple_list)))
                    next_tuple_lists.append(list(filter(lambda this_tuple: field in this_tuple, tuple_list)))
                ending_with = not ending_with
        tuple_lists = next_tuple_lists
    return flatten(tuple_lists)


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


# remove data points which use boosts we that aren't in the given list
def filter_data_points(fields_to_filter, data_points):
    return list(filter(lambda data_point: not any(
        field not in fields_to_filter and data_point['boost_vals'][field] > 0 for field in
        data_point['boost_vals'].keys()),
                       data_points))


def get_fields_without_data(fields_to_verify, data_points):
    field_set = set(fields_to_verify)
    for data_point in data_points:
        for field in list(field_set):
            if data_point['boost_vals'][field] > 0:
                field_set.remove(field)
                if len(field_set) == 0:
                    return []
    return list(field_set)


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


def apply(base, term, boost_amts):
    xp = base
    for multiplicative_group in term:
        summed_boosts = sum(boost_amts[boost] for boost in multiplicative_group["boosts"])
        xp = xp + (xp if multiplicative_group["type"] == "partial" else base) * summed_boosts // 1000
    return xp


def get_xp(base, model, boost_amts):
    base = apply(base, model['base'], boost_amts)
    additive = apply(base, model['additive'], boost_amts) - base
    multiplicative = apply(base, model['multiplicative'], boost_amts)
    bonus_xp = multiplicative * apply(boost_amts['bxp'], model['bonus'], boost_amts) // 1000
    total = additive + multiplicative + bonus_xp
    return total, total - base


def get_single_generation_of_successors(model, field, filtered_terms=[]):
    for key in filter(lambda term: term not in filtered_terms, model.keys()):
        term = model[key]
        for index in range(len(term) + 1):
            if index == 0:
                c = copy.deepcopy(model)
                c[key].insert(index, {"type": "partial", "boosts": [field]})
                if len(c[key]) > 1:
                    c[key][index + 1]["type"] = "partial"
                    yield c
                    c = copy.deepcopy(c)
                    c[key][index + 1]["type"] = "base"
                    yield c
                else:
                    yield c
            else:
                for group_type in ["partial", "base"]:
                    c = copy.deepcopy(model)
                    c[key].insert(index, {"type": group_type, "boosts": [field]})
                    yield c
        for index in range(len(term)):
            c = copy.deepcopy(model)
            c[key][index]["boosts"].append(field)
            yield c


# pass a string or list of strings to fields to generate all possible models based on those fields
# add dry mode for counting enumerations
def get_successors(starting_model, fields, filtered_terms=[]):
    models = [starting_model]
    for field in fields:  # number of generations of successors
        next_successors = []
        for model in models:  # generate the next generation of successors for all models of the previous generation
            next_successors.append(get_single_generation_of_successors(model, field, filtered_terms))
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


# retrieve all fields used in model
def get_model_fields(model):
    for term in model:
        for multiplicative_group in model[term]:
            for boost in multiplicative_group["boosts"]:
                    yield boost


def inline_dict_to_constructor_string(dict_to_string):
    return_value = "dict({})".format(
        " ".join([entry[0] + "=" + (str(entry[1]) if type(entry[1]) is list else ("\"" + entry[1] + "\",").ljust(10)) for entry in dict_to_string.items()])
    )
    return return_value


# kinda silly, but formats a model to print readable and compact. if the model changes and this looks ugly just delete
def model_to_string(model, indent=0):
    return_value = "dict({})".format(
        "".join([
            "\n    {}=[{}], ".format(term, "".join(["\n        " + inline_dict_to_constructor_string(group) + ", " for group in model[term]])) for term in model
        ])
    )
    if indent == 0:
        return return_value
    else:
        return textwrap.indent(return_value, " " * indent)


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

old_model = dict(
    base=[[['vos']]],
    base2=[],
    additive=[[['yak_track'], ['scabaras'], ['prime']]],
    multiplicative=[[['wise'], ['premier'], ['torstol'], ['outfit'], ['pulse'], ['coin', 'sceptre']], [['wisdom']], [['avatar']]],
    bonus=[[['worn_cinder']], [['cinder']]]
)

sota_model = dict(
    base=[],
    additive=[
        {"type": "partial", "boosts": ['yak_track']}
    ],
    multiplicative=[
        {"type": "partial", "boosts": ['wise', 'outfit']},
        {"type": "partial", "boosts": ['wisdom']},
        {"type": "base", "boosts": ['torstol']},
        {"type": "partial", "boosts": ['avatar']},
    ],
    bonus=[],
)

test_model = dict(
    base=[
        {'type': 'partial', 'boosts': ['vos']}
    ],
    additive=[
        {"type": "partial", "boosts": ['yak_track']}
    ],
    multiplicative=[
        {"type": "partial", "boosts": ['wise', 'outfit']},
        {"type": "partial", "boosts": ['wisdom']},
        {"type": "base", "boosts": ['torstol']},
        {"type": "partial", "boosts": ['avatar']},
    ],
    bonus=[],
)

counting_model = dict(first=[])

# base xp for singular data point tests
test_base_xp = activities["idol"]
# None to show all successor models
expected = None


# number of fields greatly increases search space, >O(n^n)
# fields_to_add = ['wisdom', 'torstol', 'outfit', 'wise']
fields_to_add = ['prime', 'premier']
allowed_failures = 0
data_filename = 'data.csv'

print('Starting model:')
print(json.dumps(test_model))

print('Loading data to test successor models against:')
data_points = get_data_points(data_filename)
print('{} data points loaded from {}'.format(len(data_points), data_filename))
# get the list of fields we're modeling
tracked_fields = list(get_model_fields(test_model)) + fields_to_add + ["bxp"]
print(tracked_fields)
print("Filtering data that uses boosts we aren't currently tracking: {}".format(tracked_fields))


test_filtered_points = filter_data_points(tracked_fields, data_points)
print("{} data points remaining after filtering".format(len(test_filtered_points)))
[print(data_point) for data_point in test_filtered_points]

fields_without_data = get_fields_without_data(tracked_fields, test_filtered_points)
if len(fields_without_data) != 0:
    raise Exception("Fields {} have no data for the current filtering, data points containing only the boosts {}".format(fields_without_data, tracked_fields))

print('Generating and testing successor models by adding {}...'.format(fields_to_add))
all_successors = list(get_successors(test_model, fields_to_add))
successful_models = list(filter_models(all_successors, test_filtered_points, allowed_failures))
print("{} error free successor models, printing at most 100".format(len(successful_models)))
print("candiate_models = [\n"+",\n".join([model_to_string(model, 4) for model in successful_models[:100]]) + "]")


sota_test_filtered_points = filter_data_points(get_model_fields(sota_model), data_points)
error_points = get_contradictory_test_points(sota_model, sota_test_filtered_points)
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
    print('Printing at most 100 of the {} successor models and their results:'.format(len(to_print)))
else:
    to_print = matching_successors
    print('{} models which add {}, matching {} boosted xp for {} base xp, found. Printing at most 100:'
          .format(len(to_print), fields_to_add, expected, test_base_xp))
to_print = test_point_successors if len(matching_successors) == 0 else matching_successors
[print(format_xp_tuple(entry[0]) + str(entry[1])) for entry in to_print[:100]]
print()

# debug new successors generation
# [print(model) for model in list(get_successors(counting_model, map(str, range(3))))]
# count all possibilities (for oeis)
# [print(i, len(list(get_successors(counting_model, map(str, range(i)))))) for i in range(10)]

# field_powerset = list(powerset(['wisdom','avatar','torstol','outfit','wise']))
# # field_powerset.sort(key=lambda)
# [print(subset) for subset in field_powerset]

