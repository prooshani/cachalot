#import <Foundation/Foundation.h>

// Helper: escape a CSV field
static NSString *CSVEscape(NSString *field) {
    if (field == nil) return @"";
    BOOL needsQuoting = [field containsString:@","] ||
                        [field containsString:@"\""] ||
                        [field containsString:@"\n"] ||
                        [field containsString:@"\r"];
    if (!needsQuoting) return field;
    NSString *escaped = [field stringByReplacingOccurrencesOfString:@"\"" withString:@"\"\""];
    return [NSString stringWithFormat:@"\"%@\"", escaped];
}

// Flatten a nested dictionary into key.path -> value
static void FlattenDict(NSDictionary *dict, NSString *prefix, NSMutableDictionary *out) {
    [dict enumerateKeysAndObjectsUsingBlock:^(id key, id val, BOOL *stop) {
        NSString *newKey = prefix.length ? [NSString stringWithFormat:@"%@.%@", prefix, key] : [NSString stringWithString:key];
        if ([val isKindOfClass:[NSDictionary class]]) {
            FlattenDict(val, newKey, out);
        } else if ([val isKindOfClass:[NSArray class]]) {
            // Join arrays with a pipe separator
            NSMutableArray *parts = [NSMutableArray array];
            for (id item in val) {
                if ([item isKindOfClass:[NSDictionary class]] || [item isKindOfClass:[NSArray class]]) {
                    NSData *d = [NSJSONSerialization dataWithJSONObject:item options:0 error:nil];
                    [parts addObject:[[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding]];
                } else {
                    [parts addObject:[item description]];
                }
            }
            out[newKey] = [parts componentsJoinedByString:@"|"];
        } else if ([val isKindOfClass:[NSNull class]]) {
            out[newKey] = @"";
        } else {
            out[newKey] = [val description];
        }
    }];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s <input.json> <output.csv>\n", argv[0]);
            return 1;
        }

        NSString *inputPath  = [NSString stringWithUTF8String:argv[1]];
        NSString *outputPath = [NSString stringWithUTF8String:argv[2]];

        NSError *error = nil;
        NSData *data = [NSData dataWithContentsOfFile:inputPath options:0 error:&error];
        if (!data) {
            fprintf(stderr, "Error reading %s: %s\n", argv[1], error.localizedDescription.UTF8String);
            return 1;
        }

        id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (!json) {
            fprintf(stderr, "Invalid JSON: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        // Normalize to an array of row dictionaries
        NSArray *rows = nil;
        if ([json isKindOfClass:[NSArray class]]) {
            rows = json;
        } else if ([json isKindOfClass:[NSDictionary class]]) {
            rows = @[json];
        } else {
            fprintf(stderr, "Top-level JSON must be an object or array of objects.\n");
            return 1;
        }

        // Flatten each row
        NSMutableArray *flatRows = [NSMutableArray array];
        NSMutableArray *orderedKeys = [NSMutableArray array];
        NSMutableSet *seenKeys = [NSMutableSet set];

        for (id row in rows) {
            if (![row isKindOfClass:[NSDictionary class]]) {
                fprintf(stderr, "Skipping non-object row.\n");
                continue;
            }
            NSMutableDictionary *flat = [NSMutableDictionary dictionary];
            FlattenDict(row, @"", flat);
            [flatRows addObject:flat];
            for (NSString *k in flat.allKeys) {
                if (![seenKeys containsObject:k]) {
                    [seenKeys addObject:k];
                    [orderedKeys addObject:k];
                }
            }
        }

        // Build CSV
        NSMutableString *csv = [NSMutableString string];
        [csv appendString:[[orderedKeys map:^id(id k){ return CSVEscape(k); }] componentsJoinedByString:@","]];
        [csv appendString:@"\n"];

        for (NSDictionary *flat in flatRows) {
            NSMutableArray *cells = [NSMutableArray array];
            for (NSString *k in orderedKeys) {
                id val = flat[k];
                NSString *s = val ? [val description] : @"";
                [cells addObject:CSVEscape(s)];
            }
            [csv appendString:[cells componentsJoinedByString:@","]];
            [csv appendString:@"\n"];
        }

        if (![csv writeToFile:outputPath atomically:YES encoding:NSUTF8StringEncoding error:&error]) {
            fprintf(stderr, "Error writing %s: %s\n", argv[2], error.localizedDescription.UTF8String);
            return 1;
        }

        printf("Wrote %s (%lu rows, %lu columns)\n",
               argv[2], (unsigned long)flatRows.count, (unsigned long)orderedKeys.count);
    }
    return 0;
}
