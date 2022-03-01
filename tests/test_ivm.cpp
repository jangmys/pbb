#include <memory>

#include "../multicore/ivm/ivm.h"

#include <gtest/gtest.h>

class IVMTest : public ::testing::Test {
protected:
    void SetUp() override {
        size_ = 15;
        ivm_ = std::make_unique<ivm>(size_);
        // q1_.Enqueue(1);
        // q2_.Enqueue(2);
        // q2_.Enqueue(3);
    }
    // void TearDown() override {}

    std::unique_ptr<ivm> ivm_;
    int size_;
    // Queue<int> q0_;
    // Queue<int> q1_;
    // Queue<int> q2_;
};


// Demonstrate some basic assertions.
TEST_F(IVMTest, BasicAssertions) {
  // Expect two strings not to be equal.
  ASSERT_EQ(ivm_->posVect[0], 15);
  // Expect equality.
  // EXPECT_EQ(7 * 6, 42);
}
