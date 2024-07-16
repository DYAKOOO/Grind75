#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "uthash.h"

typedef struct {
    int key;
    UT_hash_handle hh;
} hash_table;

bool containsDuplicate(int* nums, int numsSize) {
    if (numsSize == 1) {
        return false;
    }

    hash_table *hash = NULL;
    hash_table *elem = NULL;
    bool flag = false;

    for (int i = 0; i < numsSize; i++) {
        HASH_FIND_INT(hash, &nums[i], elem);

        if (!elem) {
            elem = (hash_table *)malloc(sizeof(hash_table));
            elem->key = nums[i];
            HASH_ADD_INT(hash, key, elem);
        } else {
            flag = true;
            break;
        }
    }

    // Free up the hash table
    hash_table *tmp;
    HASH_ITER(hh, hash, elem, tmp) {
        HASH_DEL(hash, elem);
        free(elem);
    }

    return flag;
}

void test_containsDuplicate(int* nums, int numsSize, bool expected) {
    bool result = containsDuplicate(nums, numsSize);
    printf("Test: ");
    for (int i = 0; i < numsSize; i++) {
        printf("%d ", nums[i]);
    }
    printf(" -> Expected: %s, Got: %s\n", expected ? "true" : "false", result ? "true" : "false");
}

int main() {
    int test1[] = {1, 2, 3, 4};
    int test2[] = {1, 2, 3, 1};
    int test3[] = {1, 1, 1, 3, 3, 4, 3, 2, 4, 2};
    int test4[] = {1};

    test_containsDuplicate(test1, 4, false);
    test_containsDuplicate(test2, 4, true);
    test_containsDuplicate(test3, 10, true);
    test_containsDuplicate(test4, 1, false);

    return 0;
}

/*
    Time: O(n)
    Space: O(1)
*/

// typedef struct {
// 	int key;
// 	UT_hash_handle hh; // Makes this structure hashable
// } hash_table;

// hash_table *hash = NULL, *elem, *tmp;

// bool containsDuplicate(int* nums, int numsSize){
//     if (numsSize == 1) {
//       return false;
//     }
    
//     bool flag = false;
    
//     for (int i=0; i<numsSize; i++) {
//         HASH_FIND_INT(hash, &nums[i], elem);
        
//         if(!elem) {
//            elem = malloc(sizeof(hash_table));
//            elem->key = nums[i];
//            HASH_ADD_INT(hash, key, elem);
            
//            flag = false;
//        }
//        else {
//            flag = true;
//            break;
//        }
//     }
                          
//     // Free up the hash table 
// 	HASH_ITER(hh, hash, elem, tmp) {
// 		HASH_DEL(hash, elem); free(elem);
// 	}
    
//     return flag;
// }