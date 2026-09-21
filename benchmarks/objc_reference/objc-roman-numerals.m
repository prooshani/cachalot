#import <Foundation/Foundation.h>
static NSString *Roman(int n) {
    int v[] = {1000,900,500,400,100,90,50,40,10,9,5,4,1};
    const char *s[] = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
    NSMutableString *r = [NSMutableString string];
    for (int i = 0; i < 13; i++) while (n >= v[i]) { [r appendString:@(s[i])]; n -= v[i]; }
    return r;
}
int main(void) {
    @autoreleasepool {
        int ns[] = {4, 9, 49, 58, 1994, 2024};
        for (int i = 0; i < 6; i++) printf("%d %s\n", ns[i], [Roman(ns[i]) UTF8String]);
    }
    return 0;
}
