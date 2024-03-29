import copy
import csv
import json
import textwrap
from itertools import chain, combinations, product, zip_longest
from collections import defaultdict
import random
import time
from functools import cache, wraps


def possibly_nested_value_dispatcher(possibly_nested_possibly_hashable_value):
    if isinstance(possibly_nested_possibly_hashable_value, list):
        return possibly_nested_list_to_hashable(possibly_nested_possibly_hashable_value)
    elif isinstance(possibly_nested_possibly_hashable_value, dict):
        return possibly_nested_dict_to_hashable(possibly_nested_possibly_hashable_value)
    elif isinstance(possibly_nested_possibly_hashable_value, set):
        return possibly_nested_set_to_hashable(possibly_nested_possibly_hashable_value)
    elif not hasattr(possibly_nested_possibly_hashable_value, "__hash__"):
        raise Exception("Unhashable type {} found in nested structure".format(type(possibly_nested_possibly_hashable_value)))
    else:
        return possibly_nested_possibly_hashable_value


def possibly_nested_list_to_hashable(possibly_nested_list):
    return tuple(
        possibly_nested_value_dispatcher(possibly_hashable_value) for possibly_hashable_value in possibly_nested_list)


# take a dict that has non hashable nested types and return the hashable nested version of it
def possibly_nested_dict_to_hashable(possibly_nested_dict):
    return frozenset((hashable_key, possibly_nested_value_dispatcher(possibly_hashable_value)) for
                     hashable_key, possibly_hashable_value in possibly_nested_dict.items())


# take a dict that has non hashable nested types and return the hashable nested version of it
def possibly_nested_set_to_hashable(possibly_nested_set):
    return frozenset(
        possibly_nested_value_dispatcher(possibly_hashable_value) for possibly_hashable_value in possibly_nested_set)


# dict hashing for caching
def hash_nested(func):
    """Transform mutable nested structure
    Into immutable
    Useful to be compatible with cache
    """
    class HDict(dict):
        def __hash__(self):
            return hash(possibly_nested_value_dispatcher(self))

    class HSet(set):
        def __hash__(self):
            return hash(possibly_nested_value_dispatcher(self))

    class HList(list):
        def __hash__(self):
            return hash(possibly_nested_value_dispatcher(self))

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([HDict(arg) if isinstance(arg, dict) else HSet(arg) if isinstance(arg, set) else HList(arg) if isinstance(arg, list) else arg for arg in args])
        kwargs = {k: HDict(v) if isinstance(v, dict) else HSet(v) if isinstance(v, set) else HList(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped


def inclusive_range(start, stop, step):
    return range(start, stop+step, step)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def count_iterable(it):
    return sum(1 for dummy in it)


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


# collect the boost names that appear in the data but not in the provided list
def get_data_points_boost_complement(fields_to_complement, data_points):
    appearing_fields = set()
    for data_point in data_points:
        for boost_name, boost_val in data_point['boost_vals'].items():
            if boost_val > 0:
                appearing_fields.add(boost_name)
    return list(appearing_fields - set(fields_to_complement))



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
def test_data_point(model, point, allowed_tolerance=0):
    row_xp = calculate_data_point_xp(model, point)
    return abs(point['xp_vals']['xp'] - row_xp[0]) <= allowed_tolerance and (
                point['xp_vals']['(bonus)'] == 0 or abs(point['xp_vals']['(bonus)'] - row_xp[1]) <= allowed_tolerance)


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


# get all nonzero boost values for a nonnested term
def get_non_nested_term_nonzero_boost_vals(model, boost_vals, term_name):
    return map(lambda boost_name: boost_vals[boost_name],
               filter(lambda boost_name: boost_vals[boost_name] > 0,
                      flatten(model[term_name])))


def apply(base, term, boost_vals):
    xp = base
    for multiplicative_group in term:
        summed_boosts = sum(boost_vals[boost] for boost in multiplicative_group)
        xp = xp + xp * summed_boosts // 1000
    return xp


def get_xp(base, model, boost_vals):
    constant_groups = [sum(map(lambda boost: boost_vals[boost], group)) for group in model["constant"]]
    base += sum(constant_groups)
    mitosis_percent_bases = [sum(map(lambda boost: boost_vals[boost], group)) * base // 1000 for group in model["mitosis_percent"]]
    mitosis_constant_pre_bases = [sum(map(lambda boost: boost_vals[boost], group)) for group in model["mitosis_pre_constant"]]
    mitosis_constant_post_bases = [sum(map(lambda boost: boost_vals[boost], group)) for group in model["mitosis_post_constant"]]
    base_tuples = [((percent or 0), (pre_constant or 0), (post_constant or 0)) for percent, pre_constant, post_constant
                   in
                   zip_longest(mitosis_percent_bases, mitosis_constant_pre_bases, mitosis_constant_post_bases)]
    if len(base_tuples) == 0:
        base_tuples = [(0, 0, 0)]
    base_tuples[0] = (base_tuples[0][0] + base, base_tuples[0][1], base_tuples[0][2])
    base_tuples = list(filter(lambda base_tuple: max(base_tuple) > 0, base_tuples))
    outer_total = 0
    outer_bonus = 0
    for i in range(len(base_tuples)):
        inner_base, inner_pre_constant, inner_post_constant = base_tuples[i]
        base = apply(inner_base + inner_pre_constant, model['base'], boost_vals) + inner_post_constant
        additive_terms = filter(lambda termname: termname.startswith("additive"), model.keys())
        additive_term_values = [apply(base, model[additive_term], boost_vals) - base for additive_term in additive_terms]
        chain_terms = filter(lambda termname: termname.startswith("chain"), model.keys())
        chain_term_values = [apply(base, model[chain_term], boost_vals) - base for chain_term in chain_terms]
        multiplicative = apply(sum(chain_term_values) + base, model['multiplicative'], boost_vals)
        bonus_xp = 0
        if min(boost_vals['bxp'], 1000) == 1000:
            bonus_xp = apply(multiplicative, model['bonus'], boost_vals)
        inner_total = sum(additive_term_values) + multiplicative + bonus_xp
        inner_bonus = inner_total - base
        outer_total += inner_total
        outer_bonus += inner_bonus
    return outer_total, outer_bonus


def get_single_generation_of_successors(model, field, filtered_terms=[]):
    # make a copy of the array with extra empties deleted
    model = copy.deepcopy(model)
    for repeated_term_type in ["additive", "chain"]:
        repeated_terms = list(filter(lambda termname: termname.startswith(repeated_term_type), model.keys()))
        for i in range(len(repeated_terms)-1, 0, -1):
            if len(model[repeated_terms[i]]) == 0 and len(model[repeated_terms[i-1]]) == 0:
                del model[repeated_terms[i]]
        repeated_terms = list(filter(lambda termname: termname.startswith(repeated_term_type), model.keys()))
        empty_count = 0
        for i in range(len(repeated_terms)):
            if len(model[repeated_terms[i]]) < 1:
                empty_count += 1
        if empty_count < 1:
            model[repeated_term_type + str(len(repeated_terms) + 1)] = []

    positional_terms = ["mitosis_percent", "mitosis_pre_constant", "mitosis_post_constant"]
    non_nested_terms = ["constant"]
    # number of separate xp drops for a given model
    max_positional_term_length = max(*map(lambda term: len(model[term]), positional_terms), 2)
    for positional_term in positional_terms:
        if max_positional_term_length > len(model[positional_term]):
            model[positional_term].append([])

    for key in filter(lambda term: term not in filtered_terms, model.keys()):
        term = model[key]
        if key not in non_nested_terms:
            for index in range(len(term) + 1):
                c = copy.deepcopy(model)
                c[key].insert(index, [field])
                yield c
            for index in range(len(term)):
                c = copy.deepcopy(model)
                c[key][index].append(field)
                yield c
        else:
            c = copy.deepcopy(model)
            if len(c[key]) < 1:
                c[key].append([])
            c[key][0].append(field)
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


def get_successors_reject_early(starting_model, fields, data_points, filtered_terms=[], allowed_errors=0,
                                allowed_tolerance=0):
    models = [starting_model]
    tracked_fields = list(get_model_fields(starting_model))
    for i in range(len(fields)):
        max = 0
        max_index = i
        for j in range(i, len(fields)):
            num = count_iterable(filter_data_points(tracked_fields + fields[:i] + [fields[j]], data_points))
            if num > max:
                max = num
                max_index = j
        print("{} data points for field: {}".format(max, fields[max_index]))
        fields[i], fields[max_index] = fields[max_index], fields[i]
    print("optimized field addition order: {}".format(fields))
    i = 1
    for field in fields:  # number of generations of successors
        next_successors = []
        tracked_fields.append(field)
        print("{} tolerable generations for second to last field".format(i))
        print("Generating all '{}' successors".format(field))
        reduced_data_points = list(filter_data_points(tracked_fields, data_points))
        i = 0
        for model in models:  # generate the next generation of successors for all models of the previous generation
            next_successors.extend(get_single_generation_of_successors(model, field, filtered_terms))
            i += 1
        models = list(filter_models(next_successors, reduced_data_points, allowed_errors, allowed_tolerance))
    return models


# test every model for every data point, and reject a model when it fails
def filter_models(models, data_points, allowed_failures=0, allowed_tolerance=0):
    for model in models:
        model_valid = True
        failures = 0
        for point in data_points:
            if not test_data_point(model, point, allowed_tolerance):
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


def boost_vals_to_string(boost_vals, indent=4):
    return json.dumps(dict(filter(lambda entry: entry[1] > 0, boost_vals.items())), indent=indent)


def term_to_program_string(variable_initialization, variable_name, term):
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


def range_to_comment_string(range):
    if len(range) < 10:
        return str(range)
    else:
        return str(range[:4] + ["..."] + range[-2:]).replace("'...'", "...")


def model_to_program(model, ranges, data_point=None):
    if not data_point:
        data_point = {"xp_vals": defaultdict(int),
                      "boost_vals": defaultdict(int)}
    variables_string = "-- all boosts represented as their marginal bonus times 1000\n"
    variables_string += "-- the logic for which boosts are mutually exclusive is not embedded into this program\n"
    variables_string += "-- so it is assumed you ensure your inputs are valid (e.g. no chaos altar and gilded altar)\n"
    variables_string += \
        "\n".join("{} = {}".format(field, data_point["boost_vals"][field]).ljust(30) + "  -- {}".format(range_to_comment_string(ranges[field])) for field in get_model_fields(model))

    base_string = term_to_program_string('base', 'base', model['base'])
    additives = list(filter(lambda term_name: term_name.startswith("additive") and model[term_name], model.keys()))
    additive_strings = "\n".join(term_to_program_string('base', additive_term, model[additive_term]) + "\n{} = {} - base".format(additive_term, additive_term) for additive_term in additives)
    additive_strings += "\nadditive = {}".format(" + ".join(additives))
    chains = list(filter(lambda term_name: term_name.startswith("chain") and model[term_name], model.keys()))
    chains_strings = "\n".join(term_to_program_string('base', chain_term, model[chain_term]) + "\n{} = {} - base".format(chain_term, chain_term)
                               for chain_term in chains)
    multiplicative_strings = term_to_program_string(" + ".join(chains + ["base"]), 'multiplicative', model['multiplicative'])
    bonus_strings = "bonus = 0\nif bxp == 1000 then\n"
    bonus_strings += textwrap.indent(
        term_to_program_string('multiplicative', 'bonus', model['bonus']) or "bonus = multiplicative", "  ")
    bonus_strings += "\nend"
    total_strings = 'total = additive + multiplicative + bonus\n' \
                    'return total, total - base'
    xpdrop_string = """function xpdrop(base)\n{}\nend""".format(textwrap.indent("""{}
{}
{}
{}
{}
{}""".format(base_string, additive_strings, chains_strings, multiplicative_strings, bonus_strings, total_strings), 4*" "))

    num_mitosis_groups = max(map(lambda term_name: len(model[term_name]),
                                 filter(lambda term_name: term_name.startswith("mitosis") and model[term_name],
                                        model.keys())))
    xpfunction_arglists = []
    for index in range(num_mitosis_groups):
        arglist = model["mitosis_pre_constant"][index] + list(map(lambda boost_name: "scale(base, {}, 1000)".format(boost_name), model["mitosis_percent"][index]))
        xpfunction_arglists.append(arglist)
    if len(xpfunction_arglists) == 0:
        xpfunction_arglists.append([])
    xpfunction_arglists[0].insert(0, "base")
    xpfunction_calls = ["xpdrop({})".format(" + ".join(xpfunction_arglists[index])) for index in range(num_mitosis_groups)]
    return_line = "    return {}, {}".format("total1", " + ".join(
        ["bonus1"] + list(flatten([["total{}".format(index), "bonus{}".format(index)] for index in range(2, num_mitosis_groups + 1)]))))

    xpfunction_string = """function xp(base)\n{}\n{}\nend""".format(textwrap.indent("\n".join(
        "total{}, bonus{} = {}".format(index + 1, index + 1, xpfunction_calls[index]) for index in
        range(num_mitosis_groups)), " "*4), return_line)

    calling_code = 'base = {}   -- activity xp\n' \
                   'total, bonus = xp(base)\n' \
                   'print("+" .. total/10 .. " xp (" .. bonus/10 .. " bonus xp)")\n' \
                   '-- given example should come out to {}'.format(data_point["xp_vals"]["base"],
                                                                   format_xp_tuple((data_point["xp_vals"]["xp"],
                                                                                    data_point["xp_vals"]["(bonus)"])))

    return """
function scale(xp, numerator, denominator)
    return xp * numerator // denominator
end

-- all xp values in the game engine are represented as "lots of tenths" integers (e.g. 383.5 xp is represented as 3835)
{}

{}

{}

{}
""".format(variables_string, xpdrop_string, xpfunction_string, calling_code)


# activity base xps stored in this manner to assist experiment generation and writing of boost mutual exclusion rules
# obviously not a complete list, feel free to add more activities if you're attempting to place a boost that has no
# valid experiments
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
    "cut_opal": 150,
    "cut_jade": 200,
    "cut_topaz": 250,
    "cut_sapphire": 500,
    "cut_emerald": 675,
    "cut_ruby": 850,
    "cut_diamond": 1075,
    "cut_dragonstone": 1375,
    "cut_onyx": 1675,
    "cut_hydrix": 1975,
}

fishing_activities = {
    "fish_trout": 500,
    "fish_salmon": 700,
    "fish_leaping_trout": 600,
    "fish_leaping_salmon": 820,
    "fish_monkfish": 1200,
    "fish_green_jellyfish": 1650,
    "fish_leaping_sturgeon": 920,
    "fish_shark": 1100,
    "fish_cavefish": 3000,
    "fish_rocktail": 3800,
    "fish_blue_jellyfish": 3900,
    "fish_small_crystal": 3100,
    "fish_medium_crystal": 3300,
    "fish_large_crystal": 3500,
    "fish_sailfish": 4000
}

mining_activities = {

}

anachronia_jadinko_activities = {
    "common_jadinko": 2000,
    "shadow_jadinko": 4730,
    "igneous_jadinko": 5000,
    "cannibal_jadinko": 5100,
    "aquatic_jadinko": 5700,
    "amphibious_jadinko": 4500,
    "carrion_jadinko": 6350,
    "diseased_jadinko": 5000,
    "camouflaged_jadinko": 5150,
    "draconic_jadinko": 5540,
}

hunter_activities = {
    "polar_kebbit": 150,
    "common_kebbit": 360,
    "feldip_weasel": 480,
    "desert_devil": 900,
    "spotted_kebbit": 1040,
    "hunt_penguin": 2500,
    "razor-backed_kebbit": 4700,
    "dark_kebbit": 990,
    "dashing_kebbit": 1560,
}

hunter_activities.update(anachronia_jadinko_activities)

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

abyss_rc_activities = {
    "air_rune": 5,
    "mind_rune": 5.5,
    "water_rune": 6,
    "earth_rune": 6.5,
    "fire_rune": 7,
    "body_rune:": 7.5,
    "cosmic_rune": 8,
    "chaos_rune": 8.5,
    "nature_rune": 9,
    "law_rune": 9.5,
    "death_rune": 10,
    "blood_rune": 10.5,
    "soul_rune": 275,
}
rc_activities = {
    "astral_rune": 8.7,
}
rc_activities.update(abyss_rc_activities)

wildy_slayer_activities = {
    "slay_abyssal_demon": 277.2,
}

slayer_tower_activities = {
    "slay_abyssal_demon": 277.2,
}

slayer_mask_activities = {
    "slay_abyssal_demon": 277.2,
}

slayer_activities = {}
slayer_activities.update(wildy_slayer_activities)
slayer_activities.update(slayer_tower_activities)
slayer_activities.update(slayer_mask_activities)


farm_plot_activities = {}
farm_activities = {}
farm_activities.update(farm_plot_activities)

activities = {}
# activities.update(wc_activities)
# activities.update(fishing_activities)
# activities.update(mining_activities)
# activities.update(summoning_activites)
# activities.update(arch_activities)
# activities.update(firemaking_activities)
# activities.update(crafting_activites)
activities.update(prayer_activities)
# activities.update(slayer_activities)
# activities.update(farm_activities)


def get_boost_on_default_dict(boost_vals):
    return dict(map(lambda boost_val_entry: (boost_val_entry[0], boost_val_entry[1] > 0), boost_vals.items()))


def get_list_of_boosts_on(boost_on_default_dict):
    return list(map(lambda boost_item: boost_item[0], filter(lambda boost_item: boost_item[1], boost_on_default_dict.items())))


def validate_mutex_set(mutex_set, inclusion_dict):
    count = 0
    for field in mutex_set:
        if inclusion_dict[field]:
            count += 1
            if count > 1:
                return False
    return True


# @hash_nested
# @cache
def validate_mutex_edge_list(boost_on, mutex_boost_edge_list):
    return all(not all(boost_on[boost] for boost in edge) for edge in mutex_boost_edge_list)


# @hash_nested
# @cache
def validate_mutex_clique_list(boost_on, mutex_boost_clique_list):
    return all(validate_mutex_set(mutex_set, boost_on) for mutex_set in mutex_boost_clique_list)


# return false if an impossible boost state is defined
def validate_boost_vals(boost_vals, mutex_boost_edge_list, mutex_boost_clique_list):
    boost_on = get_boost_on_default_dict(boost_vals)
    boosts_on = get_list_of_boosts_on(boost_on)
    if not validate_mutex_edge_list(boost_on, mutex_boost_edge_list):
        return False

    if not validate_mutex_clique_list(boost_on, mutex_boost_clique_list):
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
    # crystallise boost level specific to activities
    if boost_on["crystallise"] \
            and ((boost_vals_["crystallise"] in [500, 875] and activity_name not in list(chain(wc_activities, fishing_activities, hunter_activities)))  # fish/hunt too
                 or (boost_vals_["crystallise"] in [200, 400] and activity_name not in mining_activities)):
        return False
    if (boost_on["ectofuntus"] or boost_on["gilded_altar"] or boost_on["chaos_altar"]) and activity_name not in prayer_activities:
        return False
    if boost_on["dragon_rider"] and activity_name not in dragon_bone_activities:
        return False
    if boost_on["demonic_skull_divination"] and activity_name != "cursed_memory":
        return False
    if boost_on["demonic_skull_hunter"] and activity_name != "charming_moth":
        return False
    # if boost_on["demonic_skull_farming"] and activity_name != herb_farm_activities or flower_farm_activities:
    #     return False
    if boost_on["demonic_skull_farming"] and activity_name != farm_plot_activities:
        return False
    if boost_on["demonic_skull_runecrafting"] and activity_name not in abyss_rc_activities:
        return False
    if (boost_on["demonic_skull_agility"] or boost_on["wildy_sword"]) and activity_name not in "wilderness_agility":
        return False
    if boost_on["demonic_skull_slayer"] and activity_name not in wildy_slayer_activities:
        return False
    if boost_on["special_slayer_contract"] and activity_name not in chain(wildy_slayer_activities, slayer_tower_activities):
        return False
    if boost_on["morytania_legs_slayer"] and activity_name not in slayer_tower_activities:
        return False
    if boost_on["slayer_mask"] and activity_name not in slayer_mask_activities:
        return False
    if boost_on["slayer_codex"] and activity_name not in slayer_activities:
        return False
    if (boost_on["juju_god_potion"] or boost_on["zamoraks_favour"]) and activity_name not in anachronia_jadinko_activities:  # or others i forget
        return False
    if boost_on["enhanced_yaktwee"] and activity_name not in hunter_activities:
        return False
    if boost_on["brawlers"] and activity_name in rc_activities:  # could be others
        return False
    if boost_on["brassica"] and activity_name not in farm_plot_activities:
        return False
    if boost_on["prayer_aura"] and activity_name not in prayer_activities:
        return False
    if boost_on["skillchompa"] and activity_name not in list(chain(wc_activities, arch_activities, mining_activities)):  # div_activities
        return False
    return True


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
    demonic_skull_runecrafting=[0, 2500],
    demonic_skull_farming=[0, 200],
    demonic_skull_divination=[0, 200],
    demonic_skull_hunter=[0, 200],
    demonic_skull_agility=[*inclusive_range(0, 1960, 40)],
    demonic_skull_slayer=[0, 200],
    special_slayer_contract=[0, 200],
    div_energy=[0, 250],
    wildy_sword=[0, 50],
    morytania_legs_slayer=[0, 100],
    slayer_mask=[10, 50, 150, 250, 400, 500, 520, 600, 650, 670, 700, 730, 750, 770, 780, 800, 850, 860, 900, 910, 920,
                 930, 950],  # xp from mask equal to slayer level of monster mask represents
    slayer_codex=[*inclusive_range(0, 50, 10)],
    juju_god_potion=[0, 100],
    brassica=[0, 100],
    enhanced_yaktwee=[0, 20],
    protean_trap=[0, 500],
    skillchompa=[0, 100],
    perfect_juju=[0, 50],
    roar=[0, 250],
    runecrafting_gloves=[0, 1000],
    zamoraks_favour=[0, 100],
    bonfire=[0, 10, 20, 30, 40],
    festive=[0, 500],
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
    avatar=[50, 0, 60, 40, 30],
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
    # ranges are all wildy agility
    demonic_skull_runecrafting=[0, 2500],
    demonic_skull_farming=[0, 200],
    demonic_skull_divination=[0, 200],
    demonic_skull_hunter=[0, 200],
    demonic_skull_agility=[*inclusive_range(0, 1960, 40)],
    demonic_skull_slayer=[0, 200],
    special_slayer_contract=[0, 100],
    div_energy=[0, 250],
    wildy_sword=[0, 50],
    morytania_legs_slayer=[0, 100],
    slayer_mask=[0, 850, 10, 50, 150, 250, 400, 500, 520, 600, 650, 670, 700, 730, 750, 770, 780, 800, 860, 900, 910,
                 920, 930, 950],  # prefer abyssal demons
    slayer_codex=[0, 40, 50, 10, 20, 30],
    juju_god_potion=[0, 100],
    brassica=[0, 100],
    enhanced_yaktwee=[0, 20],
    protean_trap=[0, 500],
    skillchompa=[0, 100],
    perfect_juju=[0, 50],
    roar=[0, 250],
    runecrafting_gloves=[0, 1000],
    zamoraks_favour=[0, 100],
    bonfire=[0, 10, 20, 30, 40],
    festive=[500, 0],
)

# boost states that are currently unavailable to experimental design
invalid_boosts2 = dict(
    outfit=[40, 60],
    avatar=[60, 40, 30],  # avatar bonus only 60 or 0 if max fealty
    yak_track=[0, 100],  # can't toggle yak track
    wise=[10, 20, 30],
    bxp=[1000],
    premier=[100],
    coin=[20, 0],
    sceptre=[40, 0],
    worn_pulse=[500],
    worn_cinder=[1500],
    pulse=[100, 20, 40, 60, 80],
    cinder=[100, 20, 40, 60, 80],
    torstol=[20, 15, 10],
    wisdom=[0],
    inspire=[20],
    prayer_aura=[25, 20, 10],
    demonic_skull_agility=[range(40, 1960, 40)],  # i'm overlevelled for these
    prime=[100, 1000],  # prime not in game anymore
)

invalid_boosts = dict(
    bomb=[500],
    inspire=[20],
    yak_track=[200, 100],
    vos=[200],
    portable=[100],
    wise=[40, 30, 20, 10],
    prime=[100, 1000],
    bxp=[1000],
    cinder=[100, 20, 40, 60, 80],
    worn_cinder=[1500],
    pulse=[100, 20, 40, 60, 80],
    worn_pulse=[500],
    coin=[10, 20],
    sceptre=[20, 40],
    wisdom=[25],
    scabaras=[100],
    premier=[100],
    avatar=[60, 40, 30],
    torstol=[5, 20, 15, 10],
    outfit=[10, 20, 30, 40, 50, 60],
    focus=[200],
    shared=[250],
    crystallise=[200, 400, 500, 875],  # 500(875) when wc/fish/hunt, 200(400) mining (light form)
    ectofuntus=[3000],
    gilded_altar=[1500, 2000, 2500],
    chaos_altar=[2500],
    sanctifier=[2500],
    powder=[2500],
    prayer_aura=[15, 25, 20, 10],
    dragon_rider=[1000],
    # ranges are all wildy agility
    demonic_skull_runecrafting=[2500],
    demonic_skull_farming=[200],
    demonic_skull_divination=[200],
    demonic_skull_hunter=[200],
    demonic_skull_agility=[*inclusive_range(40, 1960, 40)],
    demonic_skull_slayer=[200],
    special_slayer_contract=[100],
    div_energy=[250],
    wildy_sword=[50],
    morytania_legs_slayer=[100],
    slayer_mask=[850, 10, 50, 150, 250, 400, 500, 520, 600, 650, 670, 700, 730, 750, 770, 780, 800, 860, 900, 910,
                 920, 930, 950],  # prefer abyssal demons
    slayer_codex=[40, 50, 10, 20, 30],
    juju_god_potion=[100],
    brassica=[100],
    enhanced_yaktwee=[20],
    protean_trap=[500],
    skillchompa=[100],
    perfect_juju=[50],
    roar=[250],
    runecrafting_gloves=[1000],
    zamoraks_favour=[100],
    bonfire=[10, 20, 30, 40],
)

# store the mutually exclusive boosts as an edge list for validation
# if any of these turn out to be false, both boosts need to be re-placed into the model with new experiments
# tested around April 2022, any amount of updates could render any of this untrue
mutually_exclusive_boosts_edge_list = {
    # base boosts
    ("crystallise", "portable"),  # can't crystallize a portable
    ("crystallise", "focus"),  # can't crystallise and summoning
    ("crystallise", "shared"),  # can't crystallise a div loc
    ("shared", "focus"),  # can't div loc summoning
    ("shared", "vos"),  # vos doesn't affect div locs
    ("shared", "portable"),  # can't put div locs into a portable
    ("portable", "focus"),  # you wouldn't summon a portable
    # prayer
    ("ectofuntus", "yak_track"),  # ectofuntus can't have anything
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
    ("demonic_skull_agility", "brawlers"),  # demonic skull doesn't work with at least agility brawlers
    ("demonic_skull_slayer", "morytania_legs_slayer"),  # wildy and slayer tower separate locations
    ("slayer_codex", "outfit"),  # likely the same exact boost, but needed to prove they're the same
    ("protean_trap", "crystallise"),  # can't crystallise a protean trap
    ("protean_trap", "shared"),  # can't protean trap a div loc
    ("protean_trap", "roar"),  # roar and protean trap don't stack (goes to trap i think)
}

# for compactness, some fully connected subgraphs of the mutually exclusive boost graph are represented here as sets
# each set represents a group of boosts that are totally mutually exclusive with each other
mutually_exclusive_boosts_fully_connected_subgraphs = [
    {"wisdom", "scabaras", "prime", "wisdom", "prayer_aura"},  # worn aura slot
    {"worn_cinder", "worn_pulse", "sanctifier"},  # pocket slot
    {"ectofuntus", "powder", "gilded_altar", "chaos_altar", "sanctifier"},  # prayer base multipliers
    # mutually exclusive skull effects
    {"demonic_skull_runecrafting", "demonic_skull_farming", "demonic_skull_divination", "demonic_skull_hunter",
     "demonic_skull_agility", "demonic_skull_slayer"},
]


# vos/focus order currently unknowable due to each being 20%
# vos proven to come before portable and crystallise
# perfect juju proven to come after vos, before portable
# prayer auras are all 3 proven to be in the same spot
# skillchompa proven before vos
# not sure about demonic skull farming and brassica
sota_model = dict(
    base=[["skillchompa"], ["vos"], ["crystallise", "perfect_juju"], ["portable"], ["focus"], ["shared"], ["protean_trap", "roar"],
          ["ectofuntus", "powder", "gilded_altar", "chaos_altar", "sanctifier", "dragon_rider"], ["div_energy"],
          ["demonic_skull_divination", "demonic_skull_hunter", "demonic_skull_agility", "wildy_sword"]],
    additive1=[["yak_track", "prime", "scabaras", "bomb"]],
    additive2=[["demonic_skull_runecrafting", "demonic_skull_farming", "demonic_skull_slayer", "brassica",
                "runecrafting_gloves"]],
    additive3=[["juju_god_potion"]],
    constant=[],
    chain1=[["worn_pulse"], ["pulse"], ["sceptre"], ["coin"], ["torstol"]],
    chain2=[["wise", "outfit", "premier", "inspire", "slayer_codex", "enhanced_yaktwee"], ["wisdom", "prayer_aura", "festive"], ["brawlers"]],
    chain3=[],
    multiplicative=[["avatar"]],
    bonus=[["worn_cinder"], ["cinder"]],
    mitosis_percent=[["morytania_legs_slayer"], [], ["special_slayer_contract"]],
    mitosis_pre_constant=[[], ["slayer_mask"], []],
    mitosis_post_constant=[[], [], []],
)

test_model = dict(
    base=[["skillchompa"], ["vos"], ["crystallise", "perfect_juju"], ["portable"], ["focus"], ["shared"], ["protean_trap", "roar"],
          ["ectofuntus", "powder", "gilded_altar", "chaos_altar", "sanctifier", "dragon_rider"], ["div_energy"],
          ["demonic_skull_divination", "demonic_skull_hunter", "demonic_skull_agility", "wildy_sword"]],
    additive1=[["yak_track", "prime", "scabaras", "bomb"]],
    additive2=[["demonic_skull_runecrafting", "demonic_skull_farming", "demonic_skull_slayer", "brassica",
                "runecrafting_gloves"]],
    additive3=[["juju_god_potion"]],
    constant=[],
    chain1=[["worn_pulse"], ["pulse"], ["sceptre"], ["coin"], ["torstol"]],
    chain2=[["wise", "outfit", "premier", "inspire", "slayer_codex", "enhanced_yaktwee"], ["wisdom", "prayer_aura"], ["brawlers"]],
    chain3=[],
    multiplicative=[["avatar"]],
    bonus=[["worn_cinder"], ["cinder"]],
    mitosis_percent=[["morytania_legs_slayer"], [], ["special_slayer_contract"]],
    mitosis_pre_constant=[[], [], []],
    mitosis_post_constant=[[], ["slayer_mask"], []],
)

blank_model = dict(
    base=[],
    additive1=[],
    additive2=[],
    additive3=[],
    constant=[],
    chain1=[],
    chain2=[],
    chain3=[],
    multiplicative=[],
    bonus=[],
    mitosis_percent=[],  # positional boosts that act as either base
    mitosis_pre_constant=[],
    mitosis_post_constant=[],
)

counting_model = dict(first=[])


# number of fields searched at once greatly increases search space, >A083355(n)
remaining_fields = ['firemaking_familiar', 'tangled_fishbowl', 'swift_sailfish', 'dwarven_battleaxe',
                    'brooch', 'furnace', 'sharks_tooth_necklace', 'collectors_insignia', 'fish_gloves',
                    'dragon_slayer_gloves', 'dxp', 'raf']
fields_to_add = ['festive']
allowed_errors = 0
allowed_tolerance = 0
experiment_search_time = 1
successor_generation_term_blacklist = ["mitosis_percent", "mitosis_pre_constant", "mitosis_post_constant"]
data_filename = 'data.csv'

print('Loading data to test successor models against:')
data_points = get_data_points(data_filename)
print('{} data points loaded from {}'.format(len(data_points), data_filename))
# get the list of fields we're modeling
starting_model_fields = list(get_model_fields(test_model))
tracked_fields = starting_model_fields + fields_to_add
print("Filtering data that uses boosts we aren't currently exploring:\n{}".format(tracked_fields))
test_filtered_points = list(filter_data_points(tracked_fields, data_points))
test_filtered_boosts = get_data_points_boost_complement(tracked_fields, data_points)
print("Filtered out {} points that use the fields we aren't currently exploring:\n{}".format(len(data_points) - len(test_filtered_points),
                                                                                     test_filtered_boosts))
print("{} data points remaining after filtering, showing last 10:".format(len(test_filtered_points)))
[print(data_point) for data_point in test_filtered_points[-10:]]
print("\n")

print('Starting test model:')
print(model_to_string(test_model))
print("\n")


for field in fields_to_add:
    if field in starting_model_fields:
        raise Exception("Field '{}' already in starting model".format(field))

fields_without_data = get_fields_without_data(tracked_fields, test_filtered_points)
new_fields_without_data = list(set(fields_without_data).intersection(set(fields_to_add)))
# if len(new_fields_without_data) > 1 or (len(new_fields_without_data) > 0 and "bxp" not in new_fields_without_data):
#     raise Exception("New fields {} have no data for the current filtering"
#                     .format(new_fields_without_data))

print('Generating and testing successor models by adding {} to the test model...'.format(fields_to_add))
successful_models = list(
    get_successors_reject_early(test_model, fields_to_add, data_points, successor_generation_term_blacklist,
                                allowed_errors, allowed_tolerance))
# successful_models = list(filter_models(all_successors, test_filtered_points, allowed_errors, allowed_tolerance))
print("{} successor models {}, printing:".format(len(successful_models),
                                                "with less than {} errors".format(allowed_errors) if len(successful_models) > 0 else "with no errors"))
if len(successful_models) > 0:
    print("candiate_models = [\n"+",\n".join([model_to_string(model, 4) for model in successful_models]) + "]")
print("\n")

print("Now testing all applicable data points against the sota model:")
print(model_to_string(sota_model))
print("\n")

sota_fields = list(get_model_fields(sota_model))
print("Filtering data that uses boosts not in the sota model:\n{}".format(sota_fields))
sota_test_filtered_points = list(filter_data_points(sota_fields, data_points))
sota_test_filtered_boosts = get_data_points_boost_complement(sota_fields, data_points)
print("Filtered out {} points that use the fields not in the sota model:\n{}".format(len(data_points) - len(sota_test_filtered_points),
                                                                                     sota_test_filtered_boosts))
error_points = get_contradictory_test_points(sota_model, sota_test_filtered_points)
print('{}/{} contradictory samples calculated with sota model, printing...'.format(len(error_points), len(sota_test_filtered_points)))
for error_point_dict in error_points:
    xp_vals = error_point_dict['xp_vals']
    discrepancy = format_xp_tuple(xp_vals['discrepancy'])
    calculated = format_xp_tuple(xp_vals['calculated_xp'])
    observed = format_xp_tuple((xp_vals['xp'], xp_vals['(bonus)']))
    print('discrepancy: {} calculated: {} observed: {} base: {}, boosts:{}'
          .format(discrepancy, calculated, observed, xp_vals['base'], boost_vals_to_string(error_point_dict['boost_vals']), indent=0))
print("\n")


# debug new successors generation
# [print(model) for model in list(get_successors(counting_model, map(str, range(3))))]
# count all possibilities (for oeis)
# [print(i, len(list(get_successors(counting_model, map(str, range(i)))))) for i in range(10)]

fields_for_powerset = []
field_powerset = list(powerset(fields_for_powerset))
print("Generating all subsets of {} to help with experiment design:".format(fields_for_powerset))
[print(subset) for subset in minimize_transitions(field_powerset, fields_for_powerset)]
print("\n")


def get_experiment_sampler(to_prove, candidate_models, boost_iterables, invalid_boost_values,
                           mutually_exclusive_boosts_edge_list,
                           mutually_exclusive_boosts_fully_connected_subgraphs,
                           activities):
    tracked_boosts = set()
    boosts_that_effect_placement = set()
    boosts_that_effect_placement.add("bxp")
    for model in candidate_models:
        tracked_boosts.update(get_model_fields(model))
    boosts_that_effect_placement.update(tracked_boosts)
    all_boosts = set(tracked_boosts)
    for mutex_edge in mutually_exclusive_boosts_edge_list:
        for edge_boost in mutex_edge:
            all_boosts.add(edge_boost)
        for to_prove_boost in to_prove:
            if to_prove_boost in mutex_edge:
                boosts_that_effect_placement.difference_update(mutex_edge)
    for mutex_set in mutually_exclusive_boosts_fully_connected_subgraphs:
        for set_boost in mutex_set:
            all_boosts.add(set_boost)
        for to_prove_boost in to_prove:
            if to_prove_boost in mutex_set:
                boosts_that_effect_placement.difference_update(mutex_set)
    tracked_boosts = list(tracked_boosts)
    boosts_that_effect_placement.update(to_prove)
    boosts_that_effect_placement = list(boosts_that_effect_placement)
    print("Boosts that could affect placement:\n{}".format(boosts_that_effect_placement))
    boosts_to_modulate = copy.deepcopy(boost_iterables)
    for boost in list(boosts_to_modulate.keys()):
        if boost not in tracked_boosts or boost not in boosts_that_effect_placement:
            del boosts_to_modulate[boost]
    for boost, values in invalid_boost_values.items():
        if boost in boosts_to_modulate:
            boosts_to_modulate[boost] = list(filter(lambda boost: boost not in values, boosts_to_modulate[boost]))
    print("Boosts that aren't invalid and could affect placement:\n{}".format(boosts_to_modulate))
    activities = list(activities.items())
    print("Finding first sample:")
    i = 0
    experiment_phase = False
    while True:
        i += 1
        if i % 10000 == 0:
            print("." if not experiment_phase else ":", end="")
            experiment_phase = False
        boost_entry_generator = ((boost, random.choice(boosts_to_modulate[boost])) for boost in boosts_to_modulate)
        boost_vals = dict.fromkeys(all_boosts, 0)
        boost_vals.update(boost_entry_generator)
        if not validate_boost_vals(boost_vals, mutually_exclusive_boosts_edge_list,
                                   mutually_exclusive_boosts_fully_connected_subgraphs):
            continue
        experiment_phase = True
        activity = random.choice(activities)
        experiment = dict(activity=activity, boost_vals=boost_vals)
        if not validate_experiment(experiment):
            continue
        yield experiment


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


if len(successful_models) > 1:
    print("Attempting to narrow down the {} valid candidate models by generating you an experiment".format(
        len(successful_models)))
    print("Score is in the range of {}-0, with lower being better".format(len(successful_models) - 1))

    # generate random boost combos
    infinite_experiment_sampler = get_experiment_sampler(fields_to_add, successful_models, boost_preferences,
                                                         invalid_boosts, mutually_exclusive_boosts_edge_list,
                                                         mutually_exclusive_boosts_fully_connected_subgraphs,
                                                         activities)

    min_score = len(successful_models)-1
    experiment = {}
    print("Initializing the sampler...")
    next(infinite_experiment_sampler)  # prime the sampler, takes a while for large amounts of models
    print("Sampler initialized. Starting timer")
    start_time = time.time()
    for experiment_dict in infinite_experiment_sampler:
        if experiment_search_time != 0 and time.time() - start_time > experiment_search_time:
            break
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
            print("score:", score)
            print("experiment:")
            print(json.dumps(experiment["activity"]))
            print(boost_vals_to_string(experiment['boost_vals'], 1))

    if experiment:
        test_point_successors = [(get_xp(experiment["activity"][1], model, experiment['boost_vals']), model) for model in successful_models]
        print('xp values from "best" experiment')
        [print(test_point_successor) for test_point_successor in test_point_successors]
        print("Perform this experiment:")
        print(json.dumps(experiment["activity"]))
        print(boost_vals_to_string(experiment['boost_vals'], 1))
    else:
        print("No experiments found to help narrow down the {} remaining models in {} seconds".format(
            len(successful_models), experiment_search_time))
else:
    print("Number of successful models is {}, no need to generate experiments.".format(len(successful_models)))
print("\n")

# find the most complex point to display
biggest_point = max(data_points,
                    key=lambda point: len(list(filter(lambda value: value > 0, point["boost_vals"].values()))))

lua_filename = "jagex_pseudocode.lua"

if len(error_points) > 0:
    print("Number of error points: {}, is greater than 1, taking most recently added error point and attempting to "
          "remove boosts until it works.".format(
            len(error_points)))
    # remove boosts from boost_vals and test whether we get a number
    # useful for determining if we need to exclude certain boosts to make a model valid
    data_point = error_points[-1]  # usually the most recent point anyway

    with open(lua_filename, "w") as f:
        f.write(model_to_program(sota_model, general_boost_iterables, data_point))
    print("Lua version of the sota model with the latest error point written to {}".format(lua_filename))

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
else:
    print("Last recorded data point:")
    [print(data_point_to_string_with_calculation(data_point, sota_model)) for data_point in data_points[-1:]]
    print()
    with open(lua_filename, "w") as f:
        f.write(model_to_program(sota_model, general_boost_iterables, biggest_point))
    print("Lua version of the sota model with the most complex data point written to {}".format(lua_filename))

