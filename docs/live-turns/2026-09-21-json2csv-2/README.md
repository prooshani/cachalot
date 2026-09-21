# The second Objective-C JSON-to-CSV turn, 2026-09-21 — 0.9.2 at a 52 GiB budget

The third live coding turn this project has put through a compiler, and the third to fail. It is kept
because the *second* defect in it is the interesting one: the program is correct enough to compile after a
one-line repair, run to completion and produce a valid CSV, and it still contradicts its own
documentation — which is a class of failure the corpus gate cannot see, because the gate does not run what
it builds.

| file | what it is |
|---|---|
| `json2csv.m` | the reply exactly as the model emitted it |
| `compile.txt` | `clang -fobjc-arc -framework Foundation`, one error |
| `repair.patch` | the one-line repair |
| `json2csv-repaired.m` | the repaired program, which compiles clean |
| `input.json` | the example input from the model's own reply |
| `output-after-repair.csv` | what the repaired program actually writes for it |

## Defect 1 — it does not compile

```
json2csv.m:98:41: error: no visible @interface for 'NSMutableArray' declares the selector 'map:'
```

`-map:` is not a Foundation method; it is a Swift idiom, or a category from a third-party collection
library, written into an `NSMutableArray` call as though Foundation declared it. Replacing it with the
four-line loop the rest of the file already uses for exactly this job compiles the program.

This is a model-level API error, the same class as the two before it — 0.7.0's turn declared a helper
`NSString *` and handed it `id` values, 0.9.0's sent `-stringValue` to an `NSString` twice — and it is not
this runtime's defect class. The defects this runtime produced when it was broken were malformed
`#include` lines and in-context copy corruption, and neither appears anywhere in this reply.

## Defect 2 — repaired, it runs, and its output contradicts its own example

The reply's Notes say:

> Column order follows first-seen key order across all rows.

and its worked example claims:

```csv
name,age,address.city,address.zip,tags
```

What the repaired program actually writes for that same input:

```csv
age,address.city,address.zip,name,tags
30,Paris,75001,Alice,
25,Berlin,10115,Bob,x|y
```

The cause is three lines above the claim. `FlattenDict` fills an `NSMutableDictionary`, and the loop that
builds `orderedKeys` walks `flat.allKeys`, which has no defined order. The "first-seen" ordering the reply
promises is never implemented; the column order is whatever the dictionary hands back. The *rows* are
right, the escaping is right, the flattening is right, the arrays are joined as documented, and the program
exits 0 — so a gate that compiles and diffs nothing would pass it.

Everything else holds up: RFC 4180 quoting is correct, nested objects flatten with dot notation as
described, `NSNull` maps to an empty field, the usage text and the build line are right, and the arrays
containing objects are serialised as JSON rather than described.

## Why this turn changes what the quality gate should do

The 40-case corpus gate compiles C++ and parses Python. It contains no Objective-C, and it executes
nothing. Defect 1 argues for the first half of that — add Objective-C cases and compile them the way C++
is compiled. **Defect 2 argues for the second half**, and it is the first concrete case the project has:
the program compiles, runs, exits 0, and is still wrong in a way only running it and reading the output
reveals. See `docs/NEXT-SESSION-PROMPT.md` job 3.
