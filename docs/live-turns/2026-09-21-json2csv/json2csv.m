#import <Foundation/Foundation.h>

// Convert a JSON object/array into CSV rows.
// Assumes the JSON is an array of objects (or a single object).
// Keys from the first object are used as headers.

static NSString *CSVEscape(NSString *value) {
    if (value == nil) return @"";
    NSString *s = [value stringValue];
    BOOL needsQuoting = [s containsString:@","] ||
                        [s containsString:@"\""] ||
                        [s containsString:@"\n"] ||
                        [s containsString:@"\r"];
    if (needsQuoting) {
        NSString *escaped = [s stringByReplacingOccurrencesOfString:@"\"" withString:@"\"\""];
        return [NSString stringWithFormat:@"\"%@\"", escaped];
    }
    return s;
}

static NSString *ValueToCSVCell(id value) {
    if (value == nil || value == [NSNull null]) return @"";
    if ([value isKindOfClass:[NSString class]]) return CSVEscape(value);
    if ([value isKindOfClass:[NSNumber class]]) return CSVEscape([value stringValue]);
    if ([value isKindOfClass:[NSArray class]] || [value isKindOfClass:[NSDictionary class]]) {
        // Serialize nested structures as compact JSON
        NSData *data = [NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
        NSString *json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
        return CSVEscape(json);
    }
    return CSVEscape([value description]);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <input.json> [output.csv]\n", argv[0]);
            return 1;
        }

        NSString *inputPath = [NSString stringWithUTF8String:argv[1]];
        NSString *outputPath = (argc >= 3) ? [NSString stringWithUTF8String:argv[2]] : nil;

        NSError *error = nil;
        NSData *data = [NSData dataWithContentsOfFile:inputPath options:0 error:&error];
        if (!data) {
            fprintf(stderr, "Error reading file: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (!json) {
            fprintf(stderr, "Error parsing JSON: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        NSArray *rows = nil;
        if ([json isKindOfClass:[NSArray class]]) {
            rows = json;
        } else if ([json isKindOfClass:[NSDictionary class]]) {
            rows = @[json];
        } else {
            fprintf(stderr, "Error: top-level JSON must be an array or object.\n");
            return 1;
        }

        if (rows.count == 0) {
            fprintf(stderr, "Warning: JSON array is empty; nothing to write.\n");
            return 0;
        }

        // Collect headers from the first object, preserving insertion order.
        NSMutableArray *headers = [NSMutableArray array];
        id first = rows[0];
        if ([first isKindOfClass:[NSDictionary class]]) {
            for (id key in [first allKeys]) {
                [headers addObject:[key stringValue]];
            }
        } else {
            // Array of scalars — single column
            [headers addObject:@"value"];
        }

        NSMutableString *csv = [NSMutableString string];

        // Header row
        NSMutableArray *headerCells = [NSMutableArray array];
        for (NSString *h in headers) {
            [headerCells addObject:CSVEscape(h)];
        }
        [csv appendString:[headerCells componentsJoinedByString:@","]];
        [csv appendString:@"\n"];

        // Data rows
        for (id row in rows) {
            NSMutableArray *cells = [NSMutableArray array];
            if ([row isKindOfClass:[NSDictionary class]]) {
                for (NSString *h in headers) {
                    [cells addObject:ValueToCSVCell(row[h])];
                }
            } else {
                [cells addObject:ValueToCSVCell(row)];
            }
            [csv appendString:[cells componentsJoinedByString:@","]];
            [csv appendString:@"\n"];
        }

        if (outputPath) {
            NSError *writeError = nil;
            BOOL ok = [csv writeToFile:outputPath
                            atomically:YES
                              encoding:NSUTF8StringEncoding
                                 error:&writeError];
            if (!ok) {
                fprintf(stderr, "Error writing CSV: %s\n", writeError.localizedDescription.UTF8String);
                return 1;
            }
            printf("Wrote %lu rows to %s\n", (unsigned long)rows.count, outputPath.UTF8String);
        } else {
            printf("%s", csv.UTF8String);
        }
    }
    return 0;
}
