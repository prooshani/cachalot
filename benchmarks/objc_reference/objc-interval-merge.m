#import <Foundation/Foundation.h>
int main(void) {
    @autoreleasepool {
        NSArray *in = @[@[@8,@10], @[@1,@3], @[@15,@18], @[@2,@6], @[@17,@20], @[@6,@7]];
        NSArray *sorted = [in sortedArrayUsingComparator:^NSComparisonResult(NSArray *a, NSArray *b) {
            return [a[0] compare:b[0]];
        }];
        NSMutableArray *out = [NSMutableArray array];
        for (NSArray *iv in sorted) {
            NSArray *last = out.lastObject;
            if (last && [iv[0] integerValue] <= [last[1] integerValue]) {
                NSInteger hi = MAX([last[1] integerValue], [iv[1] integerValue]);
                out[out.count - 1] = @[last[0], @(hi)];
            } else [out addObject:iv];
        }
        for (NSArray *iv in out) printf("[%ld,%ld]\n", (long)[iv[0] integerValue], (long)[iv[1] integerValue]);
    }
    return 0;
}
