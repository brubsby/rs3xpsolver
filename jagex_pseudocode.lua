
function scale(xp, numerator, denominator)
    return xp * numerator // denominator
end

-- all boosts represented as their marginal bonus times 1000
-- the logic for which boosts are mutually exclusive is not embedded into this program
-- so it is assumed you ensure your inputs are valid (e.g. no chaos altar and gilded altar)
vos = 0                   -- [0, 200]
crystallise = 0           -- [0, 200, 400, 500, 875]
portable = 0              -- [0, 100]
focus = 0                 -- [0, 200]
shared = 0                -- [0, 250]
ectofuntus = 0            -- [0, 3000]
powder = 2500             -- [0, 2500]
gilded_altar = 0          -- [0, 1500, 2000, 2500]
chaos_altar = 0           -- [0, 2500]
sanctifier = 0            -- [0, 2500]
dragon_rider = 0          -- [0, 1000]
div_energy = 0            -- [0, 250]
demonic_skull_div = 0     -- [0, 200]
demonic_skull_hunt = 0    -- [0, 200]
demonic_skull_agil = 0    -- [0, 40, 80, 120, ..., 1920, 1960]
wildy_sword = 0           -- [0, 50]
yak_track = 0             -- [0, 100, 200]
prime = 0                 -- [0, 100, 1000]
scabaras = 100            -- [0, 100]
bomb = 0                  -- [0, 500]
demonic_skull_rc = 0      -- [0, 2500]
demonic_skull_farm = 0    -- [0, 200]
demonic_skull_slay = 0    -- [0, 200]
worn_pulse = 0            -- [0, 500]
pulse = 80                -- [0, 20, 40, 60, 80, 100]
sceptre = 0               -- [0, 20, 40]
coin = 0                  -- [0, 10, 20]
torstol = 5               -- [0, 5, 10, 15, 20]
wise = 40                 -- [0, 10, 20, 30, 40]
outfit = 60               -- [0, 10, 20, 30, 40, 50, 60]
premier = 100             -- [0, 100]
inspire = 0               -- [0, 20]
wisdom = 0                -- [0, 25]
prayer_aura = 0           -- [0, 10, 15, 20, 25]
brawlers = 0              -- [0, 500, 3000]
avatar = 60               -- [0, 30, 40, 50, 60]
worn_cinder = 1500        -- [0, 1500]
cinder = 80               -- [0, 20, 40, 60, 80, 100]
bxp = 1000                -- [0, 1000]

-- all xp values in the game engine are represented as "lots of tenths" integers (e.g. 383.5 xp is represented as 3835)
base = 720  -- activity xp
base = base + scale(base, vos, 1000)
base = base + scale(base, crystallise, 1000)
base = base + scale(base, portable, 1000)
base = base + scale(base, focus, 1000)
base = base + scale(base, shared, 1000)
base = base + scale(base, ectofuntus + powder + gilded_altar + chaos_altar + sanctifier + dragon_rider, 1000)
base = base + scale(base, div_energy, 1000)
base = base + scale(base, demonic_skull_div + demonic_skull_hunt + demonic_skull_agil + wildy_sword, 1000)
additive1 = base + scale(base, yak_track + prime + scabaras + bomb, 1000)
additive1 = additive1 - base
additive2 = base + scale(base, demonic_skull_rc + demonic_skull_farm + demonic_skull_slay, 1000)
additive2 = additive2 - base
additive = additive1 + additive2
chain1 = base + scale(base, worn_pulse, 1000)
chain1 = chain1 + scale(chain1, pulse, 1000)
chain1 = chain1 + scale(chain1, sceptre, 1000)
chain1 = chain1 + scale(chain1, coin, 1000)
chain1 = chain1 + scale(chain1, torstol, 1000)
chain1 = chain1 - base
chain2 = base + scale(base, wise + outfit + premier + inspire, 1000)
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
print("+" .. total/10 .. " xp (" .. (total - base)/10 .. " bonus xp)")
-- given example should come out to +1295.0 xp (1043.0 bonus xp),

