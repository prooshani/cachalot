# A live coding turn from 2026-09-21, compiled

`json2csv.m` is the whole of Cachalot's reply to

> Write an Objective-C command line program that converts a JSON file to CSV.

in Hamed's own chat session on 0.9.0 at a 52 GiB expert budget — 1,493 tokens at 8.53 tok/s, the fourth
turn of the session, decoded with the official chat protocol at temperature 0.6. It is stored verbatim,
including the comments, so that the compiler's verdict is a verdict on what the model actually wrote.

## It does not compile

`compile.txt` is the output of

```bash
clang -fobjc-arc -framework Foundation json2csv.m -o json2csv
```

One error: `CSVEscape` takes an `NSString *` and immediately sends it `-stringValue`, a selector `NSString`
does not declare.

## Repairing that line is not enough

Replacing line 9 with `NSString *s = value;` compiles, and the program then aborts on `input.json` — which
is the example the model itself supplied in the same reply — because the same mistake appears a second time
where the compiler cannot see it, `[key stringValue]` on an `id` returned by `-allKeys`:

```
*** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason:
'-[NSTaggedPointerString stringValue]: unrecognized selector sent to instance'
```

`repair.patch` is that one-line change, for anyone who wants to reproduce the crash.

## What this is and is not evidence of

Everything else in the program is correct: RFC 4180 quoting and quote doubling, compact-JSON serialisation
of nested values, `NSNull` handling, the array-of-scalars fallback, the usage text, the build line, the
worked example, and a closing note that correctly warns `NSDictionary` does not preserve key order.

It is **one wrong method name, made twice** — a model-level type error, the same class as the 2026-09-21
turn in `docs/HANDOFF.md` section 7.2.5 that declared a helper `NSString *` and handed it `id` values.
Neither is this runtime's defect class: the defects this runtime produced before the hyper-connection fix
were malformed `#include` lines and in-context copy corruption, and neither appears here.

The quality statement for this runtime is the 40-case corpus gate in `docs/HANDOFF.md` section 7.4.8 —
20 of 20 C++ blocks compiling, 18 of 18 Python blocks parsing, 0 of 101 malformed `#include` lines, every
column equal to a hosted FP4 and a hosted FP8 reference arm. **A hand-written turn from a live chat is not
that gate**, and two of two now say so: reading a program by eye is not the check it was being used as.
