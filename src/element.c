#include <stdio.h>
#include <stdlib.h>
#include "element.h"

// node_t related functions definition --------------------
// initialize a node
node_t *feetopo_node_init(size_t tag, double r[3])
{
    node_t *node = calloc(1, sizeof(node_t));
    node->tag = tag;   // number assigned by Gmsh
    node->r[0] = r[0]; // x-coordinate
    node->r[1] = r[1]; // y-coordinate
    node->r[2] = r[2]; // z-coordinate
    return node;       // return pointer to the node
}

// free memory allocated for a node
void feetopo_node_free(node_t *node)
{
    free(node);
}

// print a node
void feetopo_node_print(node_t *node)
{
    printf("Node(%d): (%.2g,%.2g,%.2g)\n", node->tag, node->r[0], node->r[1], node->r[2]);
}

// compare two nodes
bool feetopo_node_compare(node_t *node1, node_t *node2)
{
    return (node1->tag == node2->tag);
}

// element_t related functions definition --------------------
// free element
void feetopo_element_free(element_t *element)
{
    int i;
    for (i = 0; i < element->type->nodes; i++)
    {
        free(element->nodes[i]);
    }
    free(element->nodes);
    free(element->normal);
    free(element->type);
    free(element);
}