#import <Foundation/Foundation.h>
int main(void) {
    @autoreleasepool {
        NSArray *m = @[@[@1,@2,@3], @[@4,@5,@6]];
        NSUInteger rows = m.count, cols = [m[0] count];
        for (NSUInteger c = 0; c < cols; c++) {
            NSMutableArray *line = [NSMutableArray array];
            for (NSUInteger r = 0; r < rows; r++) [line addObject:[m[r][c] stringValue]];
            printf("%s\n", [[line componentsJoinedByString:@" "] UTF8String]);
        }
    }
    return 0;
}
