/**
 * @file   abacus_unitest.cpp
 * @author Xu Li
 * @date   10 2024
 */
#include <iostream>
#include "greedy_legalize/src/abacus_legalize_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

void test_row()
{
    double xl = 0, xh = 150;
    std::vector<double> init_x = {20, 1, 300, 25, 10, 70};
    std::vector<double> node_size_x = {10, 30, 5, 2, 20, 30};
    std::vector<double> node_size_y = {10, 10, 10, 10, 10, 10};
    std::vector<double> x = {20, 1, 300, 25, 10, 70};
    double row_height = 10;
    int num_nodes = 6;
    int num_movable_nodes = 5;
    int num_filler_nodes = 0;

    std::vector<int> row_nodes = {0, 1, 2, 3, 4, 5};
    std::vector<AbacusCluster<double> > clusters (6);
    int num_row_nodes = 6;

    abacusPlaceRowCPU(
            init_x.data(),
            node_size_x.data(),
            node_size_y.data(),
            x.data(),
            row_height,
            xl, xh,
            num_nodes,
            num_movable_nodes,
            num_filler_nodes,
            row_nodes.data(), clusters.data(), num_row_nodes
            );

    printf("sol: ");
    for (unsigned int i = 0; i < row_nodes.size(); ++i)
    {
        printf("%g, ", x[row_nodes[i]]);
    }
    printf("\n");
}

DREAMPLACE_END_NAMESPACE

int main()
{
    DREAMPLACE_NAMESPACE::test_row();
}
