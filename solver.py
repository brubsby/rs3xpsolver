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


def calculate_data_point_xp(model, point):
    return get_xp(point['xp_vals']['base'], model, point['boost_vals'])


# true if valid model for point, false otherwise
def test_data_point(model, point):
    row_xp = calculate_data_point_xp(model, point)
    return point['xp_vals']['xp'] == row_xp[0] and point['xp_vals']['(bonus)'] == row_xp[1]


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


def apply(base, term, boost_vals):
    xp = base
    for multiplicative_group in term:
        summed_boosts = sum(boost_vals[boost] for boost in multiplicative_group)
        xp = xp + xp * summed_boosts // 1000
    return xp


def get_xp(base, model, boost_vals):
    base = apply(base, model['base'], boost_vals)
    additive = apply(base, model['additive'], boost_vals) - base
    chain1 = apply(base, model['chain1'], boost_vals) - base
    chain2 = apply(base, model['chain2'], boost_vals) - base
    chain3 = apply(base, model['chain3'], boost_vals) - base
    multiplicative = apply(chain1 + chain2 + chain3 + base, model['multiplicative'], boost_vals)
    bonus_xp = 0
    if min(boost_vals.get('bxp'), 1000):
        bonus_xp = apply(multiplicative, model['bonus'], boost_vals)
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
                      "boost_vals": defaultdict(int)}
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

dragon_bone_activities = {
    "baby_dragon_bones": 300,
    "dragon_bones": 720,
    "hardened_dragon_bones": 1440,
    "dragonkin_bones": 1600,
    "frost_dragon_bones": 1800,
    "reinforced_dragon_bones": 1900,
}

bone_activities = {
    "bones": 45,
    "big_bones": 150,
    "wyvern_bones": 500,
    "dagannoth_bones": 1250,
    "airut_bones": 1325,
    "ourg_bones": 1400,
    "dinosaur_bones": 1700,
}

bone_activities.update(dragon_bone_activities)

ashes_activities = {
    "impious_ashes": 40,
    "infernal_ashes": 625,
    "tortured_ashes": 900,
    "searing_ashes": 2000,
}


prayer_activities = {}
prayer_activities.update(bone_activities)
prayer_activities.update(ashes_activities)

activities = {}
# activities.update(wc_activities)
# activities.update(fishing_activities)
# activities.update(mining_activities)
# activities.update(summoning_activites)
# activities.update(arch_activities)
# activities.update(firemaking_activities)
# activities.update(crafting_activites)
activities.update(prayer_activities)

# store the mutually exclusive boosts as an edge list for validation
# if any of these turn out to be false, both boosts need to be re-placed into the model with new experiments
# tested around April 2022, any amount of updates could render any of this untrue
mutually_exclusive_boosts_edge_list = {
    ("crystallise", "portable"),  # can't crystallize a portable
    ("crystallise", "focus"),  # can't crystallise and summoning
    ("crystallise", "shared"),  # can't crystallise a div loc
    ("shared", "focus"),  # can't div loc summoning
    ("shared", "vos"),  # vos doesn't affect div locs
    ("shared", "portable"),  # can't put div locs into a portable
    ("portable", "focus"),  # you wouldn't summon a portable
    ("ectofuntus", "yak_track"),  # ectofuntus not in prif
    ("ectofuntus", "vos"),  # ectofuntus not in prif
    ("ectofuntus", "portable"),  # ectofuntus not a portable
    ("ectofuntus", "crystallise"),  # can't crystallise the ectofuntus
    ("ectofuntus", "shared"),  # ectofuntus not a div loc
    ("ectofuntus", "premier"),  # premier turns off all ecto/altar buffs?
    ("powder", "yak_track"),  # yak track does not work with powder
    ("powder", "vos"),  # vos doesn't affect bone burial
    ("powder", "portable"),  # can't bury bones in a portable
    ("powder", "crystallise"),  # can't crystallise the bones
    ("powder", "shared"),  # bones are not a div loc
    ("gilded_altar", "yak_track"),  # yak_track doesn't work with altar
    ("gilded_altar", "vos"),  # altar not in prif
    ("gilded_altar", "portable"),  # altar is not a portable
    ("gilded_altar", "crystallise"),  # can't crystallise the altar
    ("gilded_altar", "shared"),  # altar is not a div loc
    ("chaos_altar", "yak_track"),  # yak_track doesn't work with altar
    ("chaos_altar", "vos"),  # chaos altar not in prif
    ("chaos_altar", "portable"),  # altar is not a portable
    ("chaos_altar", "crystallise"),  # can't crystallise the altar
    ("chaos_altar", "shared"),  # altar is not a div loc
    ("sanctifier", "yak_track"),  # yak_track doesn't work with sanctifier
    ("sanctifier", "vos"),  # vos doesn't affect bone burying
    ("sanctifier", "portable"),  # can't bury bones with a portable
    ("sanctifier", "crystallise"),  # can't crystallise the bones
    ("sanctifier", "shared"),  # no bone div loc
}

# for compactness, some fully connected subgraphs of the mutually exclusive boost graph are represented here as sets
# each set represents a group of boosts that are totally mutually exclusive with each other
mutually_exclusive_boosts_fully_connected_subgraphs = [
    {"wisdom", "scabaras", "prime", "wisdom", "prayer_aura"},  # worn aura slot
    {"worn_cinder", "worn_pulse", "sanctifier"},  # pocket slot
    {"ectofuntus", "powder", "gilded_altar", "chaos_altar", "sanctifier"}  # prayer base multipliers
]


# vos/focus order currently unknowable due to each being 20%
# vos proven to come before portable and crystallise
# prayer auras are all 3 proven to be in the same spot
sota_model = dict(
    base=[["vos"], ["crystallise"], ["portable"], ["focus"], ["shared"], ["ectofuntus", "powder", "gilded_altar", "chaos_altar", "sanctifier", "dragon_rider"]],
    additive=[["yak_track", "prime", "scabaras", "bomb"]],
    chain1=[["worn_pulse"], ["pulse"], ["sceptre"], ["coin"], ["torstol"]],
    chain2=[["wise", "outfit", "premier", "inspire"], ["wisdom", "prayer_aura"], ["brawlers"]],
    chain3=[],
    multiplicative=[["avatar"]],
    bonus=[["worn_cinder"], ["cinder"]],
)

test_model = dict(
    base=[["vos"], ["crystallise"], ["portable"], ["focus"], ["shared"], ["ectofuntus", "powder", "gilded_altar", "chaos_altar", "sanctifier", "dragon_rider"]],
    additive=[["yak_track", "prime", "scabaras", "bomb"]],
    chain1=[["worn_pulse"], ["pulse"], ["sceptre"], ["coin"], ["torstol"]],
    chain2=[["wise", "outfit", "premier", "inspire"], ["wisdom", "prayer_aura"], ["brawlers"]],
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
    yak_track=[0, 100, 200],
    wisdom=[0, 25],
    bxp=[0, 1000],
    premier=[0, 100],
    scabaras=[0, 100],
    prime=[0, 100, 1000],
    pulse=[*inclusive_range(0, 100, 20)],
    worn_pulse=[0, 500],
    cinder=[*inclusive_range(0, 100, 20)],
    worn_cinder=[0, 1500],
    vos=[0, 200],
    coin=[0, 10, 20],
    sceptre=[0, 20, 40],
    inspire=[0, 20],
    focus=[0, 200],
    shared=[0, 250],
    brawlers=[0, 500, 3000],
    bomb=[0, 500],
    portable=[0, 100],
    crystallise=[0, 200, 400, 500, 875],
    ectofuntus=[0, 3000],
    powder=[0, 2500],
    gilded_altar=[0, 1500, 2000, 2500],
    chaos_altar=[0, 2500],
    sanctifier=[0, 2500],
    prayer_aura=[0, *inclusive_range(10, 25, 5)],
    dragon_rider=[0, 1000],
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
    ectofuntus=[0, 3000],
    gilded_altar=[0, 1500, 2000, 2500],
    chaos_altar=[0, 2500],
    sanctifier=[0, 2500],
    powder=[0, 2500],
    prayer_aura=[0, 15, 25, 20, 10],
    dragon_rider=[0, 1000],
)

# boost states that are currently unavailable to experimental design
invalid_boosts = dict(
    # outfit=[40, 60],  # 4 piece outfit
    avatar=[50, 40, 30],  # avatar bonus only 60 or 0 if max fealty
    yak_track=[0, 100],  # can't toggle yak track
    bxp=[1000],
    # premier=[100],
    coin=[20, 0],
    sceptre=[40, 0],
    worn_pulse=[500],
    worn_cinder=[1500],
    # pulse=[100,20,40,60,80],
    # cinder=[100,20,40,60,80]
    # wisdom=[0],
    inspire=[20],
    prayer_aura=[25, 20, 10]
)


def get_boost_on_default_dict(boost_vals):
    boost_on = defaultdict(int)
    boost_on.update(
        map(lambda boost_val_entry: (boost_val_entry[0], boost_val_entry[1] > 0), boost_vals.items()))
    return boost_on



def get_list_of_boosts_on(boost_on_default_dict):
    return list(map(lambda boost_item: boost_item[0], filter(lambda boost_item: boost_item[1], boost_on_default_dict.items())))


# function to quickly test a mutex
def validate_mutex_set(mutex_set, inclusion_dict):
    count = 0
    for field in mutex_set:
        if inclusion_dict[field]:
            count += 1
            if count > 1:
                return False
    return True


# return false if an impossible boost state is defined
def validate_boost_vals(boost_vals, mutex_boost_edge_list, mutex_boost_clique_list):
    boost_on = get_boost_on_default_dict(boost_vals)
    boosts_on = get_boost_on_default_dict(boost_on)
    if not all(not all(boost_on[boost] for boost in edge) for edge in mutex_boost_edge_list):
        return False

    if not all(validate_mutex_set(mutex_set, boost_on) for mutex_set in mutex_boost_clique_list):
        return False

    # ectofuntus doesn't stack with anything at all, not even really a boost
    if boost_on['ectofuntus'] and len(boosts_on) > 1:
        return False
    return True


# return false if an impossible experiment is defined
# add more rules here if it suggests an invalid experiment due to boost/activity incompatibility
def validate_experiment(experiment):
    activity_name = experiment['activity'][0]
    boost_on = defaultdict(int)
    boost_vals_ = experiment['boost_vals']
    boost_on.update(
        map(lambda boost_val_entry: (boost_val_entry[0], boost_val_entry[1] > 0), boost_vals_.items()))
    # voice of seren can only effect yew and magic trees, summoning
    if boost_on["vos"] and activity_name not in ["wc_yew", "wc_magic", "wc_ivy"] + list(
            summoning_activites.keys()):
        return False
    # yak track doesn't apply for artefact restoration
    if boost_on["yak_track"] and activity_name in arch_activities.keys():
        return False
    # can't use summoning focus on anything other than summoning
    if boost_on["focus"] and activity_name not in summoning_activites.keys():
        return False
    if boost_on["crystallise"] \
            and ((boost_vals_["crystallise"] in [500, 875] and activity_name not in list(chain(wc_activities, fishing_activities, hunter_activities)))  # fish/hunt too
                 or (boost_vals_["crystallise"] in [200, 400] and activity_name not in mining_activities)):
        return False
    if boost_on["ectofuntus"] and activity_name not in prayer_activities:
        return False
    return True


# number of fields searched at once greatly increases search space, >A083355(n)
# ['runecrafting_gloves', 'god_potion', 'bonfire', 'dxp', 'furnace', 'brassica']
# fields_to_add = []
fields_to_add = []
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
print("{} data points remaining after filtering, showing last 10:".format(len(test_filtered_points)))
[print(data_point) for data_point in test_filtered_points[-10:]]

fields_without_data = get_fields_without_data(tracked_fields, test_filtered_points)
# if len(fields_without_data) > 1 or (len(fields_without_data) > 0 and "bxp" not in fields_without_data):
#     raise Exception("Fields {} have no data for the current filtering, data points containing only the boosts {}"
#       .format(fields_without_data, tracked_fields))

print('Generating and testing successor models by adding {}...'.format(fields_to_add))
all_successors = get_successors(test_model, fields_to_add)
successful_models = list(filter_models(all_successors, test_filtered_points, allowed_failures))
print("{} error free successor models, printing at most 100".format(len(successful_models)))
print("candiate_models = [\n"+",\n".join([model_to_string(model, 4) for model in successful_models[:100]]) + "]")


sota_fields = list(get_model_fields(sota_model))
print("Filtering data that uses boosts not in the sota model: {}".format(sota_fields + fields_to_add))
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
        boost_vals = dict(boost_value_tuples)
        yield boost_vals


def count_iterable(it):
    return sum(1 for dummy in it)


# generate all valid boost combinations given a boost to iterable of values map
# and a similar boost to value map of currently impossible values
# also pass in the per_boost_depth to only calculate combinations based on the nth most preferred boost values per boost
def generate_boost_combinations(boost_iterables, invalid_boost_values, tracked_boosts, per_boost_depth=0):
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


def data_point_to_string_with_calculation(data_point, model, indent=0):
    return_strings = data_point_to_strings(data_point)
    return_strings.insert(2, "Calculated: {}".format(format_xp_tuple(calculate_data_point_xp(model, data_point))))
    return textwrap.indent("\n".join(return_strings), " " * indent)


def data_point_to_strings(data_point):
    xp_vals = data_point['xp_vals']
    return_strings = ["Base: {} xp".format(format_xp_int(xp_vals["base"], False)),
                      "Observed: {}".format(format_xp_tuple((xp_vals["xp"], xp_vals["(bonus)"]))),
                      "Boosts: {}".format(boost_vals_to_string(data_point["boost_vals"]))]
    return return_strings


def data_point_to_string(data_point, indent=0):
    return "\n".join(data_point_to_strings(data_point))


def boost_vals_to_string(boost_vals):
    return json.dumps(dict(filter(lambda entry: entry[1] > 0, boost_vals.items())), indent=4)


if len(successful_models) > 1:
    print("Attempting to narrow down the {} valid candidate models by generating you an experiment".format(
        len(successful_models)))
    print("Score is in the range of {}-0, with lower being better".format(len(successful_models) - 1))
    # generate all possible boost levels
    boost_combinations = generate_boost_combinations(boost_preferences, invalid_boosts, tracked_fields, 3)
    # filter out boost combinations for which mutual exclusion rules have been violated
    boost_combinations = filter(lambda boost_vals: validate_boost_vals(boost_vals, mutually_exclusive_boosts_edge_list,
                                                                       mutually_exclusive_boosts_fully_connected_subgraphs),
                                boost_combinations)
    experiments_product = product(boost_combinations, activities.items())
    experiment_dicts = (dict(activity=activity, boost_vals=boost_vals) for boost_vals, activity in experiments_product)
    experiment_dicts = filter(validate_experiment, experiment_dicts)

    min_score = len(successful_models)-1
    experiment = {}
    for experiment_dict in experiment_dicts:
        boost_vals = experiment_dict["boost_vals"]
        activity_xp = experiment_dict["activity"][1]
        model_xps = {}
        for model in successful_models:
            xp_tuple = get_xp(activity_xp, model, boost_vals)
            model_xps[xp_tuple] = (1 if xp_tuple not in model_xps else (model_xps[xp_tuple] + 1))
        # a lower score means the experiments have a larger amount of different outcomes across models
        score = sum(map(lambda count: count-1, model_xps.values()))
        if score < min_score:
            experiment = experiment_dict
            min_score = score
            print("score:", score, experiment)

    if experiment:
        test_point_successors = [(get_xp(experiment["activity"][1], model, experiment['boost_vals']), model) for model in successful_models]
        print('xp values from "best" experiment')
        [print(test_point_successor) for test_point_successor in test_point_successors]
        print("Perform this experiment:")
        print(json.dumps(experiment["activity"]))
        print(boost_vals_to_string(experiment['boost_vals'], 1))
    else:
        print("No experiments found to help narrow down the {} remaining models".format(len(successful_models)))
else:
    print("Successful models is 0 or 1, no need to generate experiments. Try relaxing the constraints a bit")

# find the most complex point to display
biggest_point = max(data_points, key=lambda point: len(list(filter(lambda value: value > 0, point["boost_vals"].values()))))
print(json.dumps(biggest_point))

print("Lua version of the sota model:")
print(model_to_program(sota_model, general_boost_iterables, data_points[-1]))

print("Last recorded data point:")
[print(data_point_to_string_with_calculation(data_point, sota_model)) for data_point in data_points[-1:]]
print()

if len(error_points) > 0:
    print("Error points > 1, taking most recently added error point and attempting to remove boosts until it works.")
    # remove boosts from boost_vals and test whether we get a number
    # useful for determining if we need to exclude certain boosts to make a model valid
    data_point = error_points[-1:][0]  # usually the most recent point anyway

    print("Point in question:")
    print(data_point_to_string_with_calculation(data_point, sota_model))

    data_point_boost_vals = data_point["boost_vals"]
    boost_powerset = powerset(get_list_of_boosts_on(get_boost_on_default_dict(data_point_boost_vals)))
    valid_data_point_reductions = []
    for reduced_boost_set_names in boost_powerset:
        reduced_data_point = copy.deepcopy(data_point)
        reduced_data_point_boost_vals = reduced_data_point["boost_vals"]
        for boost_name in data_point_boost_vals.keys():
            if boost_name not in reduced_boost_set_names:
                reduced_data_point_boost_vals[boost_name] = 0
        if test_data_point(sota_model, reduced_data_point):
            valid_data_point_reductions.append(reduced_data_point)

    if len(valid_data_point_reductions) > 0:
        print("These data points are valid reductions of the most recent error point:")
        for reduced_data_point in valid_data_point_reductions:
            print(data_point_to_string_with_calculation(reduced_data_point, sota_model))
    else:
        print("No reductions of the most recent error datapoint can explain the error.")