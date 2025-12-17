# Z-Algorithm
- An efficient, linear time string matching algorithm used to find all occurances of a given pattern within a larger text
- Time Complexity: O(m + n) where m is the length of the pattern and n is the length of the text

## Components

### Z-Array
- The Z-Algorithm revolves around computing the Z-array
- Lets say we have a string of length n
- The z-array Z[0, ..., n-1] stores at each index i the length of the longest substring starting at i that is also a prefix of the string s
    - Z[i] tells us how many characters from position i onwards match with the beginning of the string

```py
def zFunction(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0

    for i in range(1, n):
        if i <= r:
            k = i - l
            
            # Case 2: reuse the previously computed value
            z[i] = min(r - i + 1, z[k])

        # Try to extend the Z-box beyond r
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        # Update the [l, r] window if extended
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1

    return z

if __name__ == "__main__":
    z = zFunction("aabxaab")
    print(" ".join(map(str, z)))
```

## How this works in pattern matching
- We take the pattern and the original string, call this text
- We combine them together using some delimiter not in the text, e.g:
- s = pattern + "$" + text
- We then compute the z-array from this string, and if any z[i] == len(pattern), we found a matching string

```py
def zFunction(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0

    for i in range(1, n):
        if i <= r:
            k = i - l

            # Case 2: reuse the previously computed value
            z[i] = min(r - i + 1, z[k])

        # Try to extend the Z-box beyond r
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        # Update the [l, r] window if extended
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1

    return z


def search(text, pattern):
    s = pattern + '$' + text
    z = zFunction(s)
    pos = []
    m = len(pattern)

    for i in range(m + 1, len(z)):
        if z[i] == m:
            
            # pattern match starts here in text
            pos.append(i - m - 1)

    return pos


if __name__ == '__main__':
    text = 'aabxaabxaa'
    pattern = 'aab'

    matches = search(text, pattern)

    for pos in matches:
        print(pos, end=' ')
```