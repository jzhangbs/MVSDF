#include "IBFS/ibfs.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
// #include <iostream>
// using std::cout;
// using std::endl;

namespace py = pybind11;

py::array_t<bool> mesh_cut(py::array_t<bool>& vertices, py::array_t<uint32_t>& edges) {
    py::buffer_info vertices_info = vertices.request();
    py::buffer_info edges_info = edges.request();
    
    bool *p_vertices = reinterpret_cast<bool*>(vertices_info.ptr);
    uint32_t *p_edges = reinterpret_cast<uint32_t*>(edges_info.ptr);
    
    uint32_t nv = vertices_info.shape[0];
    uint32_t ne = edges_info.shape[0];

    auto result = py::array_t<bool>(nv);
    py::buffer_info result_info = result.request();
    bool *p_result = reinterpret_cast<bool*>(result_info.ptr);

    IBFSGraph g(IBFSGraph::IB_INIT_FAST);
    g.initSize(nv, ne);

    for (uint32_t i=0; i<nv; i++) {
        if (p_vertices[i]) g.addNode(i, 1, 0);
        else g.addNode(i, 0, 1);
    }

    for (uint32_t i=0; i<ne; i++) {
        uint32_t v1 = p_edges[3*i];
        uint32_t v2 = p_edges[3*i+1];
        uint32_t cap = p_edges[3*i+2];
        g.addEdge(v1, v2, cap, cap);
    }

    g.initGraph();
    g.computeMaxFlow();
    // cout << g.getFlow() << endl;

    for (uint32_t i=0; i<nv; i++) {
        p_result[i] = (g.isNodeOnSrcSide(i)==1);
    }
    
    return result;
}

PYBIND11_MODULE(mesh_cut_ext, m) {

    m.doc() = "EIBFS mesh cut.";

    m.def("mesh_cut", &mesh_cut);
}
