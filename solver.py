import copy
import csv
import json
import textwrap
from itertools import chain, combinations, product
from collections import defaultdict


def inclusive_range(start, stop, step):
    return range(start, stop+step, step)


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
    return filter(lambda data_point: not any(
        field not in fields_to_filter and data_point['boost_vals'][field] > 0 for field in
        data_point['boost_vals'].keys()),
                       data_points)


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
        summed_boosts = sum(boost_amts[boost] for boost in multiplicative_group)
        xp = xp + xp * summed_boosts // 1000
    return xp


def get_xp(base, model, boost_amts):
    base = apply(base, model['base'], boost_amts)
    additive = apply(base, model['additive'], boost_amts) - base
    chain1 = apply(base, model['chain1'], boost_amts) - base
    chain2 = apply(base, model['chain2'], boost_amts) - base
    chain3 = apply(base, model['chain3'], boost_amts) - base
    multiplicative = apply(chain1 + chain2 + chain3 + base, model['multiplicative'], boost_amts)
    bonus_xp = 0
    if min(boost_amts.get('bxp'), 1000):
        bonus_xp = apply(multiplicative, model['bonus'], boost_amts)
    total = additive + multiplicative + bonus_xp
    return total, total - base


def get_single_generation_of_successors(model, field, filtered_terms=[]):
    for key in filter(lambda term: term not in filtered_terms, model.keys()):
        term = model[key]
        for index in range(len(term) + 1):
            c = copy.deepcopy(model)
            c[key].insert(index, [field])
            yield c
        for index in range(len(term)):
            c = copy.deepcopy(model)
            c[key][index].append(field)
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
            for boost in multiplicative_group:
                    yield boost
    yield "bxp"


def inline_dict_to_constructor_string(dict_to_string):
    return_value = "dict({})".format(
        " ".join([entry[0] + "=" + (str(entry[1]) if type(entry[1]) is list else ("\"" + entry[1] + "\",").ljust(10)) for entry in dict_to_string.items()])
    )
    return return_value


# kinda silly, but formats a model to print readable and compact. if the model changes and this looks ugly just delete
def model_to_string(model, indent=0):
    return_string = "dict(\n"
    term_strings = []
    for term_name, term in model.items():
        return_string += "    " + term_name + "=" + json.dumps(term) + ",\n"
    return_string += "\n".join(term_strings)
    return_string += ")"
    return textwrap.indent(return_string, " " * indent)


def term_to_program_string(variable_initialization, variable_name, term, ):
    if " " in variable_initialization:
        term_string = "{} = {}\n".format(variable_name, variable_initialization)
        term_string += "\n".join("{} = {} + scale({}, {}, 1000)"
                                 .format(variable_name,
                                         variable_name,
                                         variable_name,
                                         " + ".join(term[i])) for i in range(len(term)))
    else:
        term_string = "\n".join("{} = {} + scale({}, {}, 1000)"
                                .format(variable_name,
                                        variable_initialization if i == 0 else variable_name,
                                        variable_initialization if i == 0 else variable_name,
                                        " + ".join(term[i])) for i in range(len(term)))
    return term_string


def model_to_program(model, ranges, data_point=None):
    if not data_point:
        data_point = {"xp_vals": defaultdict(int),
                      "boost_amts": defaultdict(int)}
    variables_string = "-- all boosts as their differential bonus / 1000\n"
    variables_string += \
        "\n".join("{} = {}".format(field, data_point["boost_vals"][field]).ljust(18) + "  -- {}".format(str(ranges[field])) for field in get_model_fields(model))

    base_string = "base = {}".format(data_point["xp_vals"]["base"]) + "  -- activity xp\n"
    base_string += term_to_program_string('base', 'base', model['base'])
    additive_string = term_to_program_string('base', 'additive', model['additive'])
    additive_string += "\nadditive = additive - base"
    chains = list(filter(lambda term_name: term_name.startswith("chain") and model[term_name], model.keys()))
    chains_strings = "\n".join(term_to_program_string('base', chain_term, model[chain_term]) + "\n{} = {} - base".format(chain_term, chain_term)
                               for chain_term in chains)
    multiplicative_strings = term_to_program_string(" + ".join(chains + ["base"]), 'multiplicative', model['multiplicative'])
    bonus_strings = "bonus = 0\nif bxp == 1000 then\n" + textwrap.indent(term_to_program_string('multiplicative', 'bonus', model['bonus']), "  ") + "\nend"
    total_strings = 'total = additive + multiplicative + bonus\n' \
                    'print("+" .. total/10 .. " xp (" .. (total - base)/10 .. " bonus xp)")\n' \
                    '-- given example should come out to {}'.format(format_xp_tuple((data_point["xp_vals"]["xp"], data_point["xp_vals"]["(bonus)"])))
    model_string = """
-- all xp values in the game engine are represented as "lots of tenths" integers (e.g. 383.5 xp is represented as 3835)
{}
{}
{}
{}
{}
{}
""".format(base_string, additive_string, chains_strings, multiplicative_strings, bonus_strings, total_strings)

    return """
function scale(xp, numerator, denominator)
    return xp * numerator // denominator
end

{}
{}
""".format(variables_string, model_string)


# base xp amounts that might prove useful
wc_activities = {
    "wc_yew": 1750,
    "wc_magic": 2500,
    "wc_idol": 3835,
    "wc_bamboo": 2025,
    "wc_gbamboo": 6555,
    "wc_crystal": 4345,
    "wc_tree": 250,
    "wc_oak": 375,
    "wc_willow": 675,
    "wc_teak": 850,
    "wc_acadia": 920,
    "wc_maple": 1000,
    "wc_mahogany": 1250,
    "wc_eucalyptus": 1650,
    "wc_elder": 3250,
    "wc_ivy": 3325,
}

summoning_activites = {
    "steel_titan": 4352,
}

arch_activities = {
    "artefact": 366667,
}

firemaking_activities = {
    "fm_yew": 2025,
    "fm_magic": 3038,
    "fm_elder": 4500,
    "fm_maple": 1355,
    "fm_willow": 900,
    "fm_normal": 400,
    "fm_oak": 600,
    "fm_teak": 1050,
    "fm_arctic_pine": 1250,
    "fm_acadia": 1400,
    "fm_mahogany": 1575,
    "fm_eucalyptus": 1935,
    "fm_blisterwood": 3038,
    "fm_driftwood": 4540,
}

crafting_activites = {
    "cut_dragonstone": 1375,
}

fishing_activities = {

}

mining_activities = {

}

hunter_activities = {

}

activities = {}
activities.update(wc_activities)
# activities.update(fishing_activities)
# activities.update(mining_activities)
# activities.update(summoning_activites)
# activities.update(arch_activities)
# activities.update(firemaking_activities)
# activities.update(crafting_activites)

# vos/focus order currently unknowable due to each being 20%
# shared is mutex with each focus, vos, portable
# portable is mutex with each focus, shared
# vos proven to come before portable
sota_model = dict(
    base=[["vos"], ["portable"], ["focus"], ["shared"]],
    additive=[["yak_track", "prime", "scabaras", "bomb"]],
    chain1=[["worn_pulse"], ["pulse"], ["sceptre"], ["coin"], ["torstol"]],
    chain2=[["wise", "outfit", "premier", "inspire"], ["wisdom"], ["brawlers"]],
    chain3=[],
    multiplicative=[["avatar"]],
    bonus=[["worn_cinder"], ["cinder"]],
)

test_model = dict(
    base=[["vos"], ["portable"], ["focus"], ["shared"]],
    additive=[["yak_track", "prime", "scabaras", "bomb"]],
    chain1=[["worn_pulse"], ["pulse"], ["sceptre"], ["coin"], ["torstol"]],
    chain2=[["wise", "outfit", "premier", "inspire"], ["wisdom"], ["brawlers"]],
    chain3=[],
    multiplicative=[["avatar"]],
    bonus=[["worn_cinder"], ["cinder"]],
)

counting_model = dict(first=[])




# iterables that when product-ed together will produce all boost combinations
# number of boost combinations total is at most equal to the product of total levels of each boost
general_boost_iterables = dict(
    wise=[*inclusive_range(0, 40, 10)],
    torstol=[*inclusive_range(0, 20, 5)],
    outfit=[*inclusive_range(0, 60, 10)],
    avatar=[0, *inclusive_range(30, 60, 10)],
    yak_track=[*inclusive_range(0, 200, 100)],
    wisdom=[*inclusive_range(0, 25, 25)],
    bxp=[*inclusive_range(0, 1000, 1000)],
    premier=[*inclusive_range(0, 100, 100)],
    scabaras=[*inclusive_range(0, 100, 100)],
    prime=[0, *inclusive_range(100, 1000, 900)],
    pulse=[*inclusive_range(0, 100, 20)],
    worn_pulse=[*inclusive_range(0, 500, 500)],
    cinder=[*inclusive_range(0, 100, 20)],
    worn_cinder=[*inclusive_range(0, 1500, 1500)],
    vos=[*inclusive_range(0, 200, 200)],
    coin=[*inclusive_range(0, 20, 10)],
    sceptre=[*inclusive_range(0, 40, 20)],
    inspire=[*inclusive_range(0, 20, 20)],
    focus=[*inclusive_range(0, 200, 200)],
    shared=[*inclusive_range(0, 250, 250)],
    brawlers=[0, *inclusive_range(500, 3000, 2500)],
    bomb=[*inclusive_range(0, 500, 500)],
    portable=[*inclusive_range(0, 100, 100)],
    crystallise=[0, 200, 400, 500, 875],
)

# each boost lists the order in which their state is the most to least preferable, for experiment design
# this would be filled differently for different players with different accounts, or changed as circumstances change
# ordering here also matters, as it affects the order boosts are iterated through, leave the ones you don't want to
# change at the front
boost_preferences = dict(
    bomb=[0, 500],
    inspire=[0, 20],
    brawlers=[0, 500, 3000],
    yak_track=[200, 0, 100],
    vos=[0, 200],
    portable=[0, 100],
    wise=[0, 40, 30, 20, 10],
    prime=[0, 100, 1000],
    bxp=[0, 1000],
    cinder=[0, 100, 20, 40, 60, 80],
    worn_cinder=[0, 1500],
    pulse=[0, 100, 20, 40, 60, 80],
    worn_pulse=[0, 500],
    coin=[0, 10, 20],
    sceptre=[0, 20, 40],
    wisdom=[0, 25],
    scabaras=[0, 100],
    premier=[0, 100],
    avatar=[60, 0, 50, 40, 30],
    torstol=[0, 5, 20, 15, 10],
    outfit=[0, 10, 20, 30, 40, 50, 60],
    focus=[0, 200],
    shared=[0, 250],
    crystallise=[0, 200, 400, 500, 875],  # 500(875) when wc/fish/hunt, 200(400) mining (light form)
)

# states for which I am currently unable to test for various reasons
boost_invalids = dict(
    outfit=[10, 20, 30, 40, 50, 60],  # 4 piece outfit
    avatar=[50, 40, 30],  # avatar bonus only 60 or 0 if max fealty
    yak_track=[0, 100],  # can't toggle yak track
    bxp=[1000],
    premier=[100],
    coin=[20, 0],
    sceptre=[40, 0],
    worn_pulse=[500],
    worn_cinder=[1500],
    # pulse=[100,20,40,60,80],
    # cinder=[100,20,40,60,80]
    # wisdom=[0],
    inspire=[20],
)


# return false if an impossible boost state is defined
# add more rules here if it suggests an experiment with a boost combination that's not possible (e.g. same slot)
def validate_boost_amts(boost_amts):
    boost_on = defaultdict(int)
    boost_on.update(
        map(lambda boost_amt_entry: (boost_amt_entry[0], boost_amt_entry[1] > 0), boost_amts.items()))
    # mutually exclusive (worn aura slot)
    if (boost_on["wisdom"] and boost_on["scabaras"]) or (boost_on["scabaras"] and boost_on["prime"]) or (
            boost_on["prime"] and boost_on["wisdom"]):
        return False
    # mutually exclusive (pocket slot)
    if boost_on["worn_cinder"] > 0 and boost_on["worn_pulse"] > 0:
        return False
    # can't use shared knowledge or summoning focus with portables
    if boost_on["portable"] and (boost_on["shared"] or boost_on["focus"]):
        return False
    # can't use shared knowledge with focus or vos
    if boost_on["shared"] and (boost_on["focus"] or boost_on["vos"]):
        return False
    return True


# return false if an impossible experiment is defined
# add more rules here if it suggests an invalid experiment due to boost/activity incompatibility
def validate_experiment(experiment):
    activity_name = experiment['activity'][0]
    boost_on = defaultdict(int)
    boost_amts_ = experiment['boost_amts']
    boost_on.update(
        map(lambda boost_amt_entry: (boost_amt_entry[0], boost_amt_entry[1] > 0), boost_amts_.items()))
    # voice of seren can only effect yew and magic trees, summoning
    if boost_on["vos"] and activity_name not in ["yew", "magic", "ivy"] + list(
            summoning_activites.keys()):
        return False
    # yak track doesn't apply for artefact restoration
    if boost_on["yak_track"] and activity_name in arch_activities.keys():
        return False
    # can't use summoning focus on anything other than summoning
    if boost_on["focus"] and activity_name not in summoning_activites.keys():
        return False
    if boost_on["crystallise"] \
            and ((boost_amts_["crystallise"] in [500, 875] and activity_name not in list(chain(wc_activities, fishing_activities, hunter_activities)))  # fish/hunt too
                 or (boost_amts_["crystallise"] in [200, 400] and activity_name not in mining_activities)):
        return False
    return True



run_individual_point_test = False
# base xp for singular data point tests
test_base_xp = wc_activities["wc_yew"]
# None to show all successor models
expected = None


# number of fields greatly increases search space, >A083355(n)
# ['ectofuntus', 'prayer_aura', 'dragon_gloves', 'runecrafting_gloves', 'god_potion', 'bonfire', 'dxp', 'furnace']
# fields_to_add = []
fields_to_add = ["crystallise"]
allowed_failures = 0
data_filename = 'data.csv'

print('Starting model:')
print(model_to_string(test_model))

print('Loading data to test successor models against:')
data_points = get_data_points(data_filename)
print('{} data points loaded from {}'.format(len(data_points), data_filename))
# get the list of fields we're modeling
tracked_fields = list(get_model_fields(test_model)) + fields_to_add
print("Filtering data that uses boosts we aren't currently tracking: {}".format(tracked_fields))


test_filtered_points = list(filter_data_points(tracked_fields, data_points))
print("{} data points remaining after filtering".format(len(test_filtered_points)))
[print(data_point) for data_point in test_filtered_points]

fields_without_data = get_fields_without_data(tracked_fields, test_filtered_points)
# if len(fields_without_data) > 1 or (len(fields_without_data) > 0 and "bxp" not in fields_without_data):
#     raise Exception("Fields {} have no data for the current filtering, data points containing only the boosts {}".format(fields_without_data, tracked_fields))

print('Generating and testing successor models by adding {}...'.format(fields_to_add))
all_successors = get_successors(test_model, fields_to_add)
successful_models = list(filter_models(all_successors, test_filtered_points, allowed_failures))
print("{} error free successor models, printing at most 100".format(len(successful_models)))
print("candiate_models = [\n"+",\n".join([model_to_string(model, 4) for model in successful_models[:100]]) + "]")


sota_fields = list(get_model_fields(sota_model))
print("Filtering data that uses boosts not in the sota model: {}".format(sota_fields))
sota_test_filtered_points = list(filter_data_points(sota_fields, data_points))
error_points = get_contradictory_test_points(sota_model, sota_test_filtered_points)
print('{}/{} contradictory samples calculated with sota model, printing...'.format(len(error_points), len(sota_test_filtered_points)))
for error_point_dict in error_points:
    xp_vals = error_point_dict['xp_vals']
    discrepancy = format_xp_tuple(xp_vals['discrepancy'])
    calculated = format_xp_tuple(xp_vals['calculated_xp'])
    observed = format_xp_tuple((xp_vals['xp'], xp_vals['(bonus)']))
    print('discrepancy: {} calculated: {} observed: {} base: {}, boosts:{}'
          .format(discrepancy, calculated, observed, xp_vals['base'], error_point_dict['boost_vals']))


# debug new successors generation
# [print(model) for model in list(get_successors(counting_model, map(str, range(3))))]
# count all possibilities (for oeis)
# [print(i, len(list(get_successors(counting_model, map(str, range(i)))))) for i in range(10)]

fields_for_powerset = ['avatar','torstol','outfit']
field_powerset = list(powerset(fields_for_powerset))
[print(subset) for subset in minimize_transitions(field_powerset, fields_for_powerset)]

# next up, experiment suggester
# first, generate candidate models that could explain all data with new boosts (we already do this)
# next, generate all possible sets of boosts and activity xps, these are all possible experiments
# test the candidate models for all experiments
# choose the experiment that produces the widest array of different xp values for all models
# i.e. choose the experiment that minimizes the largest number of models producing the same xp
# if there are multiple experiments which produce optimal results, sort by how annoying it is to do an experiment
# if all experiments produce one value for all models, something is wrong


def boost_iterables_to_boost_value_tuples(boost_iterables):
    for boost, levels in boost_iterables.items():
        yield [(boost, level) for level in levels]


def boost_value_product_to_boost_vals_dict(boost_value_product):
    for boost_value_tuples in boost_value_product:
        boost_amts = dict(boost_value_tuples)
        if validate_boost_amts(boost_amts):
            yield boost_amts


def count_iterable(it):
    return sum(1 for dummy in it)


# generate all valid boost combinations given a boost to iterable of values map
# and a similar boost to value map of currently impossible values
# also pass in the per_boost_depth to only calculate combinations based on the nth most preferred boost values per boost
def generate_valid_boost_combinations(boost_iterables, invalid_boost_values, tracked_boosts, per_boost_depth=0):
    boosts_less_invalids = copy.deepcopy(boost_iterables)
    for boost in list(boosts_less_invalids.keys()):
        if boost not in tracked_boosts:
            del boosts_less_invalids[boost]
    for boost, values in invalid_boost_values.items():
        if boost in boosts_less_invalids:
            boosts_less_invalids[boost] = list(filter(lambda boost: boost not in values, boosts_less_invalids[boost]))
    for boost in tracked_boosts:
        if per_boost_depth > 0:
            boosts_less_invalids[boost] = boosts_less_invalids[boost][:per_boost_depth]
    boost_value_product = product(*boost_iterables_to_boost_value_tuples(boosts_less_invalids))
    return boost_value_product_to_boost_vals_dict(boost_value_product)


if len(successful_models) > 1:
    print("Attempting to narrow down the {} valid candidate models by generating you an experiment".format(len(successful_models)))
    print("Score is in the range of {}-0, with lower being better".format(len(successful_models)-1))
    # generate all possible boost levels
    boost_combinations = generate_valid_boost_combinations(boost_preferences, boost_invalids, tracked_fields, 2)
    experiments_product = product(boost_combinations, activities.items())
    experiment_dicts = (dict(activity=activity, boost_amts=boost_amts) for boost_amts, activity in experiments_product)
    experiment_dicts = filter(validate_experiment, experiment_dicts)

    min_score = len(successful_models)-1
    experiment = {}
    for experiment_dict in experiment_dicts:
        boost_amts = experiment_dict["boost_amts"]
        activity_xp = experiment_dict["activity"][1]
        model_xps = {}
        for model in successful_models:
            xp_tuple = get_xp(activity_xp, model, boost_amts)
            model_xps[xp_tuple] = (1 if xp_tuple not in model_xps else (model_xps[xp_tuple] + 1))
        # a lower score means the experiments have a larger amount of different outcomes across models
        score = sum(map(lambda count: count-1, model_xps.values()))
        if score < min_score:
            experiment = experiment_dict
            min_score = score
            print("score:", score, experiment)

    if experiment:
        test_point_successors = [(get_xp(experiment["activity"][1], model, experiment['boost_amts']), model) for model in successful_models]
        print('xp values from "best" experiment')
        [print(test_point_successor) for test_point_successor in test_point_successors]
        print("Perform this experiment:")
        print(json.dumps(experiment["activity"]))
        print(json.dumps(dict(filter(lambda entry: entry[1] > 0, experiment['boost_amts'].items())), indent=1))
    else:
        print("No experiments found to help narrow down the {} remaining models".format(len(successful_models)))
else:
    print("Successful models is 0 or 1, no need to generate experiments. Try relaxing the constraints a bit")

# find the most complex point to display
biggest_point = max(data_points, key=lambda point: len(list(filter(lambda value: value > 0, point["boost_vals"].values()))))
print(json.dumps(biggest_point))

print("Lua version of the sota model:")
print(model_to_program(sota_model, general_boost_iterables, biggest_point))
