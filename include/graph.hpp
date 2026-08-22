#pragma once
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

// ---------------------------------------------------------------------------
// A minimal "IR": every op is a typed node with input node indices.
// The Graph just stores nodes in the order they were added; because every
// node's inputs must already exist, insertion order IS topological order.
// This mirrors (at toy scale) how a real tensor compiler represents a
// forward pass before scheduling/optimizing it.
// ---------------------------------------------------------------------------

enum class OpType {
    Input,      // leaf: data supplied by the caller
    MatMul,     // C = A @ B
    AddBias,    // C = A + bias (broadcast over rows)
    ReLU,       // C = max(A, 0)
    FusedBiasReLU // C = max(A + bias, 0)  -- operator fusion of AddBias + ReLU
};

struct Shape {
    int rows;
    int cols;
};

struct Node {
    OpType op;
    std::vector<int> inputs;   // indices into Graph::nodes
    Shape shape;               // output shape of this node
    std::string name;
};

class Graph {
public:
    int addInput(const std::string& name, Shape shape) {
        nodes.push_back(Node{OpType::Input, {}, shape, name});
        return (int)nodes.size() - 1;
    }

    int addMatMul(int a, int b, const std::string& name) {
        Shape sa = nodes.at(a).shape;
        Shape sb = nodes.at(b).shape;
        if (sa.cols != sb.rows)
            throw std::runtime_error("MatMul shape mismatch in node " + name);
        nodes.push_back(Node{OpType::MatMul, {a, b}, {sa.rows, sb.cols}, name});
        return (int)nodes.size() - 1;
    }

    int addBias(int a, int bias, const std::string& name) {
        nodes.push_back(Node{OpType::AddBias, {a, bias}, nodes.at(a).shape, name});
        return (int)nodes.size() - 1;
    }

    int addReLU(int a, const std::string& name) {
        nodes.push_back(Node{OpType::ReLU, {a}, nodes.at(a).shape, name});
        return (int)nodes.size() - 1;
    }

    // operator-fusion entry point: same math as AddBias -> ReLU, one node
    int addFusedBiasReLU(int a, int bias, const std::string& name) {
        nodes.push_back(Node{OpType::FusedBiasReLU, {a, bias}, nodes.at(a).shape, name});
        return (int)nodes.size() - 1;
    }

    void print() const {
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& n = nodes[i];
            printf("[%2zu] %-14s shape=(%d,%d) inputs=", i, opName(n.op), n.shape.rows, n.shape.cols);
            for (int in : n.inputs) printf("%d ", in);
            printf(" name=%s\n", n.name.c_str());
        }
    }

    static const char* opName(OpType op) {
        switch (op) {
            case OpType::Input: return "Input";
            case OpType::MatMul: return "MatMul";
            case OpType::AddBias: return "AddBias";
            case OpType::ReLU: return "ReLU";
            case OpType::FusedBiasReLU: return "FusedBiasReLU";
        }
        return "?";
    }

    std::vector<Node> nodes;
};
