#import <Foundation/Foundation.h>
@interface LRU : NSObject
- (instancetype)initWithCapacity:(NSUInteger)n;
- (NSNumber *)get:(NSString *)k;
- (void)put:(NSString *)k value:(NSNumber *)v;
@end
@implementation LRU { NSUInteger _cap; NSMutableArray<NSString *> *_order; NSMutableDictionary<NSString *, NSNumber *> *_map; }
- (instancetype)initWithCapacity:(NSUInteger)n { if ((self = [super init])) { _cap = n; _order = [NSMutableArray array]; _map = [NSMutableDictionary dictionary]; } return self; }
- (void)touch:(NSString *)k { [_order removeObject:k]; [_order addObject:k]; }
- (NSNumber *)get:(NSString *)k { NSNumber *v = _map[k]; if (v) [self touch:k]; return v; }
- (void)put:(NSString *)k value:(NSNumber *)v {
    if (!_map[k] && _map.count == _cap) { NSString *old = _order.firstObject; [_order removeObjectAtIndex:0]; [_map removeObjectForKey:old]; }
    _map[k] = v; [self touch:k];
}
@end
int main(void) {
    @autoreleasepool {
        LRU *c = [[LRU alloc] initWithCapacity:2];
        [c put:@"a" value:@1]; [c put:@"b" value:@2];
        printf("get a: %s\n", [[[c get:@"a"] description] UTF8String] ?: "miss");
        [c put:@"c" value:@3];
        printf("get b: %s\n", [[[c get:@"b"] description] UTF8String] ?: "miss");
        [c put:@"d" value:@4];
        printf("get a: %s\n", [[[c get:@"a"] description] UTF8String] ?: "miss");
        printf("get c: %s\n", [[[c get:@"c"] description] UTF8String] ?: "miss");
        printf("get d: %s\n", [[[c get:@"d"] description] UTF8String] ?: "miss");
    }
    return 0;
}
