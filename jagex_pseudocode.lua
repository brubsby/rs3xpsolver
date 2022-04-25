
function scale(xp, numerator, denominator)
    return xp * numerator // denominator
end

-- all xp values in the game engine are represented as "lots of tenths" integers (e.g. 383.5 xp is represented as 3835)
-- all boosts represented as their marginal bonus times 1000
-- the logic for which boosts are mutually exclusive is not embedded into this program
-- so it is assumed you ensure your inputs are valid (e.g. no chaos altar and gilded altar)
skillchompa = 0                 -- [0, 100]
vos = 0                         -- [0, 200]
crystallise = 0                 -- [0, 200, 400, 500, 875]
perfect_juju = 0                -- [0, 50]
portable = 0                    -- [0, 100]
focus = 0                       -- [0, 200]
shared = 0                      -- [0, 250]
protean_trap = 0                -- [0, 500]
roar = 0                        -- [0, 250]
ectofuntus = 0                  -- [0, 3000]
powder = 2500                   -- [0, 2500]
gilded_altar = 0                -- [0, 1500, 2000, 2500]
chaos_altar = 0                 -- [0, 2500]
sanctifier = 0                  -- [0, 2500]
dragon_rider = 0                -- [0, 1000]
div_energy = 0                  -- [0, 250]
demonic_skull_divination = 0    -- [0, 200]
demonic_skull_hunter = 0        -- [0, 200]
demonic_skull_agility = 0       -- [0, 40, 80, 120, ..., 1920, 1960]
wildy_sword = 0                 -- [0, 50]
yak_track = 0                   -- [0, 100, 200]
prime = 0                       -- [0, 100, 1000]
scabaras = 100                  -- [0, 100]
bomb = 0                        -- [0, 500]
demonic_skull_runecrafting = 0  -- [0, 2500]
demonic_skull_farming = 0       -- [0, 200]
demonic_skull_slayer = 0        -- [0, 200]
brassica = 0                    -- [0, 100]
runecrafting_gloves = 0         -- [0, 1000]
juju_god_potion = 0             -- [0, 100]
worn_pulse = 0                  -- [0, 500]
pulse = 80                      -- [0, 20, 40, 60, 80, 100]
sceptre = 0                     -- [0, 20, 40]
coin = 0                        -- [0, 10, 20]
torstol = 5                     -- [0, 5, 10, 15, 20]
wise = 40                       -- [0, 10, 20, 30, 40]
outfit = 60                     -- [0, 10, 20, 30, 40, 50, 60]
premier = 100                   -- [0, 100]
inspire = 0                     -- [0, 20]
slayer_codex = 0                -- [0, 10, 20, 30, 40, 50]
enhanced_yaktwee = 0            -- [0, 20]
wisdom = 0                      -- [0, 25]
prayer_aura = 0                 -- [0, 10, 15, 20, 25]
brawlers = 0                    -- [0, 500, 3000]
avatar = 60                     -- [0, 30, 40, 50, 60]
worn_cinder = 1500              -- [0, 1500]
cinder = 80                     -- [0, 20, 40, 60, 80, 100]
morytania_legs_slayer = 0       -- [0, 100]
special_slayer_contract = 0     -- [0, 200]
slayer_mask = 0                 -- [10, 50, 150, 250, ..., 930, 950]
bxp = 1000                      -- [0, 1000]

function xpdrop(base)
    base = base + scale(base, skillchompa, 1000)
    base = base + scale(base, vos, 1000)
    base = base + scale(base, crystallise + perfect_juju, 1000)
    base = base + scale(base, portable, 1000)
    base = base + scale(base, focus, 1000)
    base = base + scale(base, shared, 1000)
    base = base + scale(base, protean_trap + roar, 1000)
    base = base + scale(base, ectofuntus + powder + gilded_altar + chaos_altar + sanctifier + dragon_rider, 1000)
    base = base + scale(base, div_energy, 1000)
    base = base + scale(base, demonic_skull_divination + demonic_skull_hunter + demonic_skull_agility + wildy_sword, 1000)
    additive1 = base + scale(base, yak_track + prime + scabaras + bomb, 1000)
    additive1 = additive1 - base
    additive2 = base + scale(base, demonic_skull_runecrafting + demonic_skull_farming + demonic_skull_slayer + brassica + runecrafting_gloves, 1000)
    additive2 = additive2 - base
    additive3 = base + scale(base, juju_god_potion, 1000)
    additive3 = additive3 - base
    additive = additive1 + additive2 + additive3
    chain1 = base + scale(base, worn_pulse, 1000)
    chain1 = chain1 + scale(chain1, pulse, 1000)
    chain1 = chain1 + scale(chain1, sceptre, 1000)
    chain1 = chain1 + scale(chain1, coin, 1000)
    chain1 = chain1 + scale(chain1, torstol, 1000)
    chain1 = chain1 - base
    chain2 = base + scale(base, wise + outfit + premier + inspire + slayer_codex + enhanced_yaktwee, 1000)
    chain2 = chain2 + scale(chain2, wisdom + prayer_aura, 1000)
    chain2 = chain2 + scale(chain2, brawlers, 1000)
    chain2 = chain2 - base
    multiplicative = chain1 + chain2 + base
    multiplicative = multiplicative + scale(multiplicative, avatar, 1000)
    bonus = 0
    if bxp == 1000 then
      bonus = multiplicative + scale(multiplicative, worn_cinder, 1000)
      bonus = bonus + scale(bonus, cinder, 1000)
    end
    total = additive + multiplicative + bonus
    return total, total - base
end

function xp(base)
    total1, bonus1 = xpdrop(base + scale(base, morytania_legs_slayer, 1000))
    total2, bonus2 = xpdrop(slayer_mask)
    total3, bonus3 = xpdrop(scale(base, special_slayer_contract, 1000))
    return total1, bonus1 + total2 + bonus2 + total3 + bonus3
end

base = 720   -- activity xp
total, bonus = xp(base)
print("+" .. total/10 .. " xp (" .. bonus/10 .. " bonus xp)")
-- given example should come out to +1295.0 xp (1043.0 bonus xp),
