#

def scale(xp, numerator, denominator):
    return xp * numerator // denominator

vos = 0
wise = 0
outfit = 10
wisdom = 25
torstol = 5
avatar = 60
pulse = 100
premier = 100

base_xp = 366667
base_xp = base_xp + scale(base_xp, vos, 1000)
xp = base_xp + scale(base_xp, wise + outfit + premier, 1000) + scale(base_xp, pulse, 1000)
xp = xp + scale(xp, wisdom, 1000)
xp = xp + scale(base_xp, torstol, 1000)
xp = xp + scale(xp, avatar, 1000)
print(xp, xp - base_xp)
