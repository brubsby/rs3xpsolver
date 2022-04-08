function scale(xp, numerator, denominator)
    return xp * numerator // denominator
end

-- all boosts as their differential bonus / 1000
vos = 0             -- [0, 200]
focus = 0           -- [0, 200]
shared = 0          -- [0, 250]
yak_track = 200     -- [0, 100, 200]
prime = 0           -- [0, 100, 1000]
scabaras = 100      -- [0, 100]
bomb = 0            -- [0, 500]
worn_pulse = 0      -- [0, 500]
pulse = 100         -- [0, 20, 40, 60, 80, 100]
sceptre = 20        -- [0, 20, 40]
coin = 10           -- [0, 10, 20]
torstol = 0         -- [0, 5, 10, 15, 20]
wise = 40           -- [0, 10, 20, 30, 40]
outfit = 50         -- [0, 10, 20, 30, 40, 50, 60]
premier = 100       -- [0, 100]
inspire = 0         -- [0, 20]
wisdom = 0          -- [0, 25]
brawlers = 0        -- [0, 500, 3000]
avatar = 50         -- [0, 30, 40, 50, 60]
worn_cinder = 0     -- [0, 1500]
cinder = 100        -- [0, 20, 40, 60, 80, 100]
bxp = 0             -- [0, 1000]

-- all xp values in the game engine are represented as "lots of tenths" integers (e.g. 383.5 xp is represented as 3835)
base = 1750  -- activity xp
base = base + scale(base, vos, 1000)
base = base + scale(base, focus, 1000)
base = base + scale(base, shared, 1000)
additive = base + scale(base, yak_track + prime + scabaras + bomb, 1000)
additive = additive - base
chain1 = base + scale(base, worn_pulse, 1000)
chain1 = chain1 + scale(chain1, pulse, 1000)
chain1 = chain1 + scale(chain1, sceptre, 1000)
chain1 = chain1 + scale(chain1, coin, 1000)
chain1 = chain1 + scale(chain1, torstol, 1000)
chain1 = chain1 - base
chain2 = base + scale(base, wise + outfit + premier + inspire, 1000)
chain2 = chain2 + scale(chain2, wisdom, 1000)
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
-- given example should come out to +295.4 xp (120.4 bonus xp), 