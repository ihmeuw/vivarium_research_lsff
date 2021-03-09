def add_commas(n):
    """
    input: n
    -----
    output: str(n), formatted with commas (are these called commas?)
    -----
    example: add_commas(1000)='1,000'
    """
    n = int(float(n))
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return str(n)[:-3] + ',' + str(n)[-3:]
    if n < 1_000_000_000:
        return str(n)[:-6] + ',' + str(n)[-6:-3] + ',' + str(n)[-3:]
    else:
        raise Exception('n >= 1 billion!')
        
def round_ticks(x):
    """
    input: n
    -----
    output: n rounded DOWN for cleaner numbers in tick display
    -----
    caution: absurdly imprecise; just for visuals
    """
    x = int(x)
    if x < 10:
        return x
    if x < 100:
        return x - (x % 10)
    if x < 1_000:
        return x - (x % 100)
    if x < 1_000_000:
        return x - (x % 500)
    if x < 10_000_000:
        return x - (x % 10_000)