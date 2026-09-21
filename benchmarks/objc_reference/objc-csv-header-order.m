#import <Foundation/Foundation.h>
static NSString *Esc(NSString *s) {
    if ([s rangeOfCharacterFromSet:[NSCharacterSet characterSetWithCharactersInString:@",\"\n"]].location == NSNotFound) return s;
    return [NSString stringWithFormat:@"\"%@\"", [s stringByReplacingOccurrencesOfString:@"\"" withString:@"\"\""]];
}
int main(void) {
    @autoreleasepool {
        NSArray *rows = @[
            @[@[@"name", @"Alice"], @[@"age", @"30"], @[@"city", @"Paris"]],
            @[@[@"name", @"Bob"], @[@"note", @"says \"hi\", twice"], @[@"age", @"25"]],
            @[@[@"zip", @"10115"], @[@"name", @"Cy"]],
        ];
        NSMutableArray *cols = [NSMutableArray array];
        for (NSArray *row in rows)
            for (NSArray *kv in row)
                if (![cols containsObject:kv[0]]) [cols addObject:kv[0]];
        NSMutableArray *head = [NSMutableArray array];
        for (NSString *c in cols) [head addObject:Esc(c)];
        printf("%s\n", [[head componentsJoinedByString:@","] UTF8String]);
        for (NSArray *row in rows) {
            NSMutableDictionary *d = [NSMutableDictionary dictionary];
            for (NSArray *kv in row) d[kv[0]] = kv[1];
            NSMutableArray *cells = [NSMutableArray array];
            for (NSString *c in cols) [cells addObject:Esc(d[c] ?: @"")];
            printf("%s\n", [[cells componentsJoinedByString:@","] UTF8String]);
        }
    }
    return 0;
}
