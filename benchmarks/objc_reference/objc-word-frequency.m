#import <Foundation/Foundation.h>
int main(void) {
    @autoreleasepool {
        NSString *text = @"the cat and the hat and THE bat sat on a mat with the cat";
        NSCountedSet *set = [NSCountedSet set];
        for (NSString *w in [[text lowercaseString] componentsSeparatedByString:@" "])
            if (w.length) [set addObject:w];
        NSArray *words = [[set allObjects] sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
            NSUInteger ca = [set countForObject:a], cb = [set countForObject:b];
            if (ca != cb) return ca > cb ? NSOrderedAscending : NSOrderedDescending;
            return [a compare:b];
        }];
        for (NSString *w in words) printf("%s %lu\n", [w UTF8String], (unsigned long)[set countForObject:w]);
    }
    return 0;
}
