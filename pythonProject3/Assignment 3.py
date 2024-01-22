# CSCI 323 Winter
# Assignment 3
# Fariha T Khan
# Geeks for geeks

import random
import string

import pandas as pd
import time
import matplotlib.pyplot as plt


# random text
def random_text(m):
    from lorem_text import lorem
    text = " "
    for i in range(m):
        x = random.choice(text)
        text = text + x
        return text.upper()


# random pattern
def random_pattern(n, text):
    m = len(text)
    idx = random.randint(0, (len(text) - n))
    return text[idx: idx + n]


# Native
def python_search(txt, pat):
    x = txt.find(pat)


# Brute Force
def bf_search(pat, txt):
    M = len(pat)
    N = len(txt)

    # A loop to slide pat[] one by one */
    for i in range(N - M + 1):
        j = 0

        # For current index i, check
        # for pattern match */
        while j < M:
            if txt[i + j] != pat[j]:
                break
            j += 1

        if j == M:
            print("Pattern found at index ", i)


# Knuth-Morris-Pratt
def kmp_search(pat, txt):
    M = len(pat)
    N = len(txt)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0] * M
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    compute_lpsarray(pat, M, lps)

    i = 0  # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            print("Found pattern at index " + str(i - j))
            j = lps[j - 1]

        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1


def compute_lpsarray(pat, M, lps):
    len = 0  # length of the previous longest prefix suffix

    lps[0]  # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len - 1]

                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1


def rk_search(pat, txt, q):
    d = 256
    M = len(pat)
    N = len(txt)
    i = 0
    j = 0
    p = 0  # hash value for pattern
    t = 0  # hash value for txt
    h = 1
    q = 101

    # The value of h would be "pow(d, M-1)%q"
    for i in range(M - 1):
        h = (h * d) % q

    # Calculate the hash value of pattern and first window
    # of text
    for i in range(M):
        p = (d * p + ord(pat[i])) % q
        t = (d * t + ord(txt[i])) % q

    # Slide the pattern over text one by one
    for i in range(N - M + 1):
        # Check the hash values of current window of text and
        # pattern if the hash values match then only check
        # for characters on by one
        if p == t:
            # Check for characters one by one
            for j in range(M):
                if txt[i + j] != pat[j]:
                    break
                else:
                    j += 1

            # if p == t and pat[0...M-1] = txt[i, i+1, ...i+M-1]
            if j == M:
                print("Pattern found at index " + str(i))

        # Calculate hash value for next window of text: Remove
        # leading digit, add trailing digit
        if i < N - M:
            t = (d * (t - ord(txt[i]) * h) + ord(txt[i + M])) % q

            # We might get negative values of t, converting it to
            # positive
            if t < 0:
                t = t + q

    rk_search(pat, txt, q)


def plot_times_line_graph(dict_searches):
    for search in dict_searches:
        x = dict_searches[search].keys()
        y = dict_searches[search].values()
        plt.plot(x, y, label=search)
        plt.legend()
        plt.title("Run time of Search Algorithms")
        plt.xlabel("Number of Elements")
        plt.ylabel("Time for 100 trials")
        plt.savefig("line_graph.png")
        plt.show()


def plot_times_bar_graph(dict_searches, sizes, searches):
    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for search in searches:
        search_num += 1
        d = dict_searches[search.__name__]
        x_axis = [j + 0.05 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.07, alpha=.25, label=search.__name__)

    plt.legend()
    plt.title("Run time of Search Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 trials")
    plt.savefig("bar_graph.png")
    plt.show()


def main():
    max_int = 1000
    trials = 10
    #text generated from lorem
    text = "Quibusdam quia accusamus quasi reprehenderit commodi quae vitae hic, inventore dolorum sapiente harum in, " \
           "reprehenderit accusamus corrupti cupiditate ducimus delectus voluptatum perspiciatis ex accusantium " \
           "repellendus porro, consectetur nostrum harum molestiae repudiandae cupiditate tenetur assumenda inventore " \
           "esse autem odit, quas quia eaque atque esse quod animi velit quidem a libero.Dolore rem optio, " \
           "deleniti eveniet incidunt maiores corporis. Dignissimos officia consectetur p erspiciatis placeat " \
           "molestiae ab sed illo blanditiis laboriosam neque, eos asperiores fugiat prov ident natus dignissimos " \
           "voluptate perspiciatis eaque voluptatibus laboriosam, fugit sapiente in obcaecati libero adipisci. "
    dict_searches = {}
    searches = [bf_search, kmp_search, rk_search, python_search]
    for search in searches:
        dict_searches[search.__name__] = {}
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for size in sizes:
        for search in searches:
            dict_searches[search.__name__][size] = 0
        for trial in range(1, trials):
            m = len(text)
            arr = random_text(m)
            n = len(arr)
            pattern = random_pattern(n, arr)
            idx = random.randint(1, size) - 1

            for search in searches:
                start_time = time.time()
                end_time = time.time()
                net_time = end_time - start_time
                dict_searches[search.__name__][size] += net_time * 1000

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame.from_dict(dict_searches).T
    print(df)

    # plot_times_line_graph(dict_searches)
    plot_times_bar_graph(dict_searches, sizes, searches)


if __name__ == '__main__':
    main()
