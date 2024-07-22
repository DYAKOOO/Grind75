package main

import "fmt"

func containsDuplicate(nums []int) bool {
	if len(nums) <= 1 {
		return false
	}

	xm := make(map[int]struct{})

	for _, v := range nums {
		if _, ok := xm[v]; ok {
			return true
		}

		xm[v] = struct{}{}
	}

	return false
}

func main() {
	nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 1}
	fmt.Println(containsDuplicate(nums)) // Output: true
}