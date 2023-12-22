#include <stdlib.h>
#include <stdbool.h>

// these are Gmsh's nomenclature for element types
#define ELEMENT_TYPE_UNDEFINED 0
#define ELEMENT_TYPE_LINE2 1
#define ELEMENT_TYPE_TRIANGLE3 2
#define ELEMENT_TYPE_QUADRANGLE4 3
#define ELEMENT_TYPE_TETRAHEDRON4 4
#define ELEMENT_TYPE_PRISM6 6
#define ELEMENT_TYPE_POINT1 15

typedef struct node_t node_t;
typedef struct element_t element_t;
typedef struct element_type_t element_type_t;
typedef struct entity_t entity_t;
typedef struct physical_group_t physical_group_t;

// mesh-related structs ------------------------
// node
struct node_t
{
    size_t tag;  // number assigned by Gmsh
    double r[3]; // spatial coordinates of the node
};

// element type
struct element_type_t
{
    unsigned int id;             // Gmsh's id for the element type
    unsigned int dim;            // i.e. 0:0D, 1:1D, 2:2D, 3:3D
    unsigned int order;          // polynomial order of the element
    unsigned int nodes;          // total, i.e. 2 for line2
    unsigned int vertices;       // the corner nodes, i.e. 4 for tet10
    unsigned int faces;          // i.e. number of neighbors
    unsigned int nodes_per_face; // (max) number of nodes per face
};

// element
struct element_t
{
    size_t tag; // number assigned by Gmsh
    double volume;
    double area;
    double *normal;                   // outward normal direction (only for 2d elements)
    element_type_t *type;             // pointer to the element type
    entity_t *entity;                 // pointer to the entity this element belongs to
    physical_group_t *physical_group; // pointer to the physical group this element belongs to
    node_t **nodes;                   // pointer to the nodes, node[j] points to the j-th node
};

// node_t related functions declaration --------------------
node_t *feetopo_node_init(size_t tag, double r[3]);
void feetopo_node_free(node_t *node);
void feetopo_node_print(node_t *node);
bool feetopo_node_compare(node_t *node1, node_t *node2);

// element_t related functions declaration --------------------
element_t *feetopo_point_init(size_t tag, node_t *node);
element_t *feetopo_line2_init(size_t tag, node_t **nodes);
element_t *feetopo_quad4_init(size_t tag, node_t **nodes);

void feetopo_point_print(element_t *this, bool verbose);
void feetopo_line2_print(element_t *this, bool verbose);
void feetopo_quad4_print(element_t *this, bool verbose);

node_t *feetopo_element_get_common_nodes(element_t *this, element_t *other);
size_t *feetopo_element_get_common_nodes_tags(element_t *this, element_t *other);

void feetopo_element_free(element_t *element);
void feetopo_element_add_entity(element_t *element, entity_t *entity);
void feetopo_element_add_physical_group(element_t *element, physical_group_t *physical_group);
