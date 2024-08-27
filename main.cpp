#include "Game.h"

// Define grid size
const int GRID_WIDTH = 60 * 4;
const int GRID_HEIGHT = 40 * 4;
const int CELL_SIZE = 4; // Each grid cell will be 4x4 pixels

const float CELSIUS_TO_KELVIN = 273.15f;


// Define particle types
enum class ParticleType {
    EMPTY,
    SAND,
    WATER,
    METHANE,
    FIRE,
    SMOKE,
    STEAM,
    STONE,
    DUST,
    LAVA,
    CLONE,
    ICE,
    PLASMA,
    WALL,
    DIAMOND,
    MERCURY,
    OIL,
    ERASER,
    WOOD,
    BURNING_WOOD,
    COUNT // Use COUNT to represent the number of items
};

enum class ParticleState {
    EMPTY,
    SOLID,
    POWDER,
    FLUID,
    GAS,
    PLASMA,
};

// Function to sample from a range based on weighted probabilities
int sampleFromProbabilities(const std::vector<float>& probabilities) {
    if (probabilities.empty()) return -1; // Early exit if the input is empty

    // Step 1: Compute the total sum of probabilities
    float totalProbability = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);

    // Step 2: Generate a random number within the range of the total probability
    float randomValue = RNG<float>::getRange(0.0f, totalProbability);

    // Step 3: Iterate through the probabilities to find the correct index
    float cumulativeSum = 0.0f;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulativeSum += probabilities[i];
        if (randomValue <= cumulativeSum) {
            return static_cast<int>(i);
        }
    }

    return -1; // Should never reach here if probabilities are correctly normalized
}

// Softmax function for normalization
std::vector<float> altsoftmax(const std::vector<float>& weights) {
    std::vector<float> probabilities(weights.size());

    float sum = 0.0f;

    // Calculate exponentials and sum them up
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i];
    }

    // Normalize by dividing by the sum
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] > 0) {
            probabilities[i] = weights[i] / sum;
        }
        else {
            probabilities[i] = 0.0f;  // Assign zero probability for zero weight
        }
    }

    return probabilities;
}

float smoothstep(float edge0, float edge1, float x) {
    // Scale, bias and saturate x to 0..1 range
    x = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    // Evaluate polynomial
    return x * x * (3 - 2 * x);
}

std::vector<float> interpolateWeights(ParticleState state, float density) {
    std::vector<std::vector<float>> weights;

    // Define weight templates
    std::vector<std::vector<float>> solid_weights = {
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Extremely light materials
        { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Very light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f }, // Light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Neutrally buoyant materials
        { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Slightly dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f }, // Very dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f }  // Extremely dense materials
    };
    std::vector<std::vector<float>> powder_weights = {
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Extremely light materials
        { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Very light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f }, // Light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Neutrally buoyant materials
        { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Slightly dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f }, // Very dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f }  // Extremely dense materials
    };
    std::vector<std::vector<float>> fluid_weights = {
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Extremely light materials
        { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Very light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f }, // Light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Neutrally buoyant materials
        { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Slightly dense materials
        { 0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 1.0f, 1.0f, 1.0f }, // Very dense materials
        { 0.0f, 0.0f, 0.0f, 0.0001f, 0.0001f, 1.0f, 0.1f, 0.1f }  // Extremely dense materials
    };
    std::vector<std::vector<float>> gas_weights = {
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Extremely light materials
        { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Very light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f }, // Light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Neutrally buoyant materials
        { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Slightly dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f }, // Very dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f }  // Extremely dense materials
    };
    std::vector<std::vector<float>> plasma_weights = {
        { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Extremely light materials
        { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, // Very light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f }, // Light materials
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Neutrally buoyant materials
        { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, // Slightly dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f }, // Very dense materials
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f }  // Extremely dense materials
    };

    if (state == ParticleState::SOLID) {
        weights = solid_weights;
    }
    else if (state == ParticleState::POWDER) {
        weights = powder_weights;
    }
    else if (state == ParticleState::FLUID) {
        weights = fluid_weights;
    }
    else if (state == ParticleState::GAS) {
        weights = gas_weights;
    }
    else if (state == ParticleState::PLASMA) {
        weights = plasma_weights;
    }

    // Initialize finalWeights with zeros
    std::vector<float> finalWeights(8, 0.0f);

    // Interpolation logic based on density
    if (density < 0.01f) { // Extremely light materials
        finalWeights = weights[0];
    }
    else if (density < 0.25f) { // Very light materials
        float t = smoothstep(0.01f, 0.25f, density);
        for (size_t i = 0; i < finalWeights.size(); ++i) {
            finalWeights[i] = (1 - t) * weights[0][i] + t * weights[1][i];
        }
    }
    else if (density < 1.0f) { // Light materials
        float t = smoothstep(0.25f, 1.0f, density);
        for (size_t i = 0; i < finalWeights.size(); ++i) {
            finalWeights[i] = (1 - t) * weights[1][i] + t * weights[2][i];
        }
    }
    else if (density < 1.2f) { // Neutrally buoyant materials
        float t = smoothstep(1.0f, 1.2f, density);
        for (size_t i = 0; i < finalWeights.size(); ++i) {
            finalWeights[i] = (1 - t) * weights[2][i] + t * weights[3][i];
        }
    }
    else if (density < 1.4f) { // Neutrally buoyant materials
        float t = smoothstep(1.2f, 1.4f, density);
        for (size_t i = 0; i < finalWeights.size(); ++i) {
            finalWeights[i] = (1 - t) * weights[3][i] + t * weights[4][i];
        }
    }
    else if (density < 1000.0f) { // Slightly dense materials
        float t = smoothstep(1.4f, 1000.0f, density);
        for (size_t i = 0; i < finalWeights.size(); ++i) {
            finalWeights[i] = (1 - t) * weights[4][i] + t * weights[5][i];
        }
    }
    else if (density < 2000.0f) { // Very dense materials
        float t = smoothstep(1000.0f, 2000.0f, density);
        for (size_t i = 0; i < finalWeights.size(); ++i) {
            finalWeights[i] = (1 - t) * weights[5][i] + t * weights[6][i];
        }
    }
    else { // Extremely dense materials
        finalWeights = weights[6];
    }

    return finalWeights;
}

// Function to get movement directions based on density
std::vector<std::pair<float, std::vector<std::pair<int, int>>>> getMovementDirectionsFromDensity(ParticleState state, float density) {
    // Define movement directions
    std::vector<std::pair<int, int>> directions = {
        {0, 1}, {-1, 1}, {1, 1}, // Upwards directions
        {-1, 0}, {1, 0},         // Horizontal directions
        {0, -1}, {-1, -1}, {1, -1} // Downwards directions
    };

    std::vector<float> finalWeights = interpolateWeights(state, density);

    // Calculate probabilities using softmax
    std::vector<float> probabilities = altsoftmax(finalWeights);

    // Create a vector to hold directions with their probabilities
    std::vector<std::tuple<int, int, float>> directionProbabilities;

    // Assign directions and probabilities to each direction
    for (size_t i = 0; i < directions.size(); ++i) {
        if (probabilities[i] > 0) {
            directionProbabilities.emplace_back(directions[i].first, directions[i].second, probabilities[i]);
        }
    }

    // Sort the directionProbabilities by probabilities in descending order
    std::sort(directionProbabilities.begin(), directionProbabilities.end(),
        [](const std::tuple<int, int, float>& a, const std::tuple<int, int, float>& b) {
            return std::get<2>(a) > std::get<2>(b);
        });

    // Group the sorted directions into tiers based on identical probabilities
    std::pair<float, std::vector<std::pair<int, int>>> currentTier;
    float currentProbability = std::get<2>(directionProbabilities[0]);


    std::vector<std::pair<float, std::vector<std::pair<int, int>>>> movementDirectionsT;
    for (const std::tuple<int, int, float>& direction : directionProbabilities) {
        float prob = std::get<2>(direction);
        if (prob == currentProbability) {
            // Same probability, add to the current tier
            currentTier.first += prob;
            currentTier.second.push_back({ std::get<0>(direction), std::get<1>(direction) });
        }
        else {
            // New probability tier, add the current tier to movementDirections and start a new tier
            movementDirectionsT.push_back(currentTier);
            currentTier.first = 0;
            currentTier.second.clear();
            currentTier.first += prob;
            currentTier.second.push_back({ std::get<0>(direction), std::get<1>(direction) });
            currentProbability = prob;
        }
    }

    // Add the last tier to movementDirections
    if (!currentTier.second.empty()) {
        movementDirectionsT.push_back(currentTier);
    }

    std::vector<std::pair<float, std::vector<std::pair<int, int>>>> movementDirections = movementDirectionsT;
    return movementDirections;
}

struct AlchemicPrerequisites {
    ParticleType type;
};

struct AlchemicResults {
    ParticleType type;
    double particleTemp = -1;
};

struct AlchemicReaction {
    double halflife; // chance per frame between 0-1 for reaction to occur
    std::vector<AlchemicPrerequisites> prerequisites;
    std::vector<AlchemicResults> results;
};

struct Emission {
    ParticleType type;
    double halflife; // chance per frame between 0-1 for emission to occur
};

struct generalParticleData {
    ParticleType type;

    std::string name;
    glm::vec4 color;

    // pos is encoded by grid[posx][posy]
    glm::vec2 velocity;
    glm::vec2 remainder; // Accumulated velocity remainder 

    double density; // Kg/m^3

    double temperature; // degrees K
    double thermalConductivity; // W/m*K  = getRoughly(data.thermalConductivity, 0.01);
    double specificHeatCapacity; // kJ/Kg*K  = getRoughly(data.specificHeatCapacity, 0.01);
    double heatReceived;  // Store the amount of heat received from neighbors

    double lowerTransitionPoint;
    ParticleType lowerTransitionType;

    double upperTransitionPoint;
    ParticleType upperTransitionType;

    double halflife; // in updates/frames
    ParticleType endOfLifeType;

    ParticleState state;

    std::vector<AlchemicReaction> reactions; // potential reactions

    std::vector<Emission> emissions; // particles to emit

    std::vector<std::pair<float, std::vector<std::pair<int, int>>>> movementDirections; // Directions to check for movement
};

float lastPrintJ = 0;

generalParticleData getParticleData(ParticleType type) {
    generalParticleData data;

    bool specificTempDetails = false;

    data.type = type;

    data.density = 0.0; // Default has no density

    data.temperature = 30 + CELSIUS_TO_KELVIN; // Default temp of 30c
    data.thermalConductivity = 1; // Default has no Thermal Conductivity
    data.specificHeatCapacity = 1; // Default has no Thermal Density
    data.heatReceived = 0.0;

    data.lowerTransitionPoint = -1; // Default to low to be obtainable
    data.upperTransitionPoint = 9999999.9; // Default to high to be obtainable

    data.halflife = -1; // Default will never timeout

    data.state = ParticleState::EMPTY;

    data.movementDirections = {}; // Default has no movement directions

    if (type == ParticleType::EMPTY) {
        data.name = "EMPTY";
        data.color = BLACK;

        data.thermalConductivity = 0; // Empty has no Thermal Conductivity
        data.specificHeatCapacity = 0; // Empty has no Thermal Density
    }
    else if (type == ParticleType::SAND) {
        data.name = "SAND";
        data.color = YELLOW;

        data.density = 1700.0;
        if (specificTempDetails) {
            data.thermalConductivity = 0.27; // Thermal conductivity for dry sand (W/m*K)
            data.specificHeatCapacity = 0.8; // Specific heat capacity for sand (kJ/kg*K)
        }
        data.state = ParticleState::POWDER;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::WATER) {
        data.name = "WATER";
        data.color = BLUE;

        data.density = 998.0;
        if (specificTempDetails) {
            data.thermalConductivity = 0.6; // Thermal conductivity of water (W/m*K)
            data.specificHeatCapacity = 4.18; // Specific heat capacity of water (kJ/kg*K)
        }
        data.lowerTransitionPoint = 0 + CELSIUS_TO_KELVIN;
        data.lowerTransitionType = ParticleType::ICE;
        data.upperTransitionPoint = 100 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::STEAM;
        data.state = ParticleState::FLUID;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::METHANE) {
        data.name = "METHANE";
        data.color = GREEN;

        data.density = 0.65;
        if (specificTempDetails) {
            data.thermalConductivity = 0.034; // Thermal conductivity of methane (W/m*K)
            data.specificHeatCapacity = 2.2; // Specific heat capacity of methane (kJ/kg*K)
        }
        data.state = ParticleState::GAS;
        data.upperTransitionPoint = 537 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::FIRE;

        AlchemicReaction reaction;
        reaction.prerequisites.push_back({ ParticleType::FIRE });
        reaction.results.push_back({ ParticleType::FIRE, 1960 + CELSIUS_TO_KELVIN });
        reaction.halflife = getRoughly(1.0 / 3.0, 0.1);
        data.reactions.push_back(reaction);

        reaction = {};
        reaction.prerequisites.push_back({ ParticleType::PLASMA });
        reaction.results.push_back({ ParticleType::FIRE, 1960 + CELSIUS_TO_KELVIN });
        reaction.halflife = getRoughly(1.0, 0.1);
        data.reactions.push_back(reaction);

        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::FIRE) {
        data.name = "FIRE";
        data.color = glm::mix(YELLOW, RED, 0.5);

        data.density = 0.3;
        if (specificTempDetails) {
            data.thermalConductivity = 90.0; // Updated value for flame thermal conductivity in W/m*K
            data.specificHeatCapacity = 1.0; // Estimated specific heat capacity for flames (kJ/kg*K)
        }
        data.halflife = getRoughly(1.0 / 300.0, 0.1);
        data.endOfLifeType = ParticleType::SMOKE;
        data.temperature = 950 + CELSIUS_TO_KELVIN;
        data.state = ParticleState::GAS;
        data.lowerTransitionPoint = 200 + CELSIUS_TO_KELVIN;
        data.lowerTransitionType = ParticleType::SMOKE;
        data.upperTransitionPoint = 7800 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::PLASMA;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);

        AlchemicReaction reaction;
        reaction.prerequisites.push_back({ ParticleType::WATER });
        reaction.results.push_back({ ParticleType::EMPTY, -1 });
        reaction.halflife = getRoughly(1.0 / 8.0, 0.1);
        data.reactions.push_back(reaction);
    }
    else if (type == ParticleType::SMOKE) {
        data.name = "SMOKE";
        data.color = GRAY;

        data.density = 1.2;
        if (specificTempDetails) {
            data.thermalConductivity = 0.01; // W/m*K (approximate for smoke)
            data.specificHeatCapacity = 1.0; // kJ/kg*K (approximate for smoke particles)
        }
        data.halflife = getRoughly(1.0 / 300.0, 0.1);
        data.endOfLifeType = ParticleType::EMPTY;
        data.upperTransitionPoint = 350 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::FIRE;
        data.state = ParticleState::GAS;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::STEAM) {
        data.color = glm::mix(GRAY, BLUE, 0.5);
        data.name = "STEAM";

        data.density = 0.6;
        if (specificTempDetails) {
            data.thermalConductivity = 0.02; // Thermal conductivity of steam (W/m*K)
            data.specificHeatCapacity = 2.0; // Specific heat capacity of steam (kJ/kg*K)
        }
        data.halflife = getRoughly(1.0 / 300.0, 0.1);
        data.endOfLifeType = ParticleType::WATER;
        data.temperature = 150 + CELSIUS_TO_KELVIN;
        data.lowerTransitionPoint = 100 + CELSIUS_TO_KELVIN;
        data.lowerTransitionType = ParticleType::WATER;
        data.upperTransitionPoint = 10000 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::PLASMA;
        data.state = ParticleState::GAS;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::STONE) {
        data.name = "STONE";
        data.color = glm::mix(GRAY, BLACK, 0.5);

        data.density = 2800.0;
        if (specificTempDetails) {
            data.thermalConductivity = 2.5; // W/m*K (average for stone)
            data.specificHeatCapacity = 0.84; // kJ/kg*K
        }
        data.upperTransitionPoint = 1500 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::LAVA;
        data.state = ParticleState::SOLID;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::DUST) {
        data.name = "DUST";
        data.color = glm::mix(YELLOW, WHITE, 0.5);

        data.density = 49.0;
        if (specificTempDetails) {
            data.thermalConductivity = 0.05; // W/m*K (approximate for dust)
            data.specificHeatCapacity = 0.8; // kJ/kg*K (similar to sand)
        }
        data.upperTransitionPoint = 350 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::FIRE;
        data.state = ParticleState::POWDER;

        AlchemicReaction reaction;
        reaction.prerequisites.push_back({ ParticleType::FIRE });
        reaction.results.push_back({ ParticleType::FIRE });
        reaction.halflife = getRoughly(1.0 / 8.0, 0.1);
        data.reactions.push_back(reaction);

        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::LAVA) {
        data.name = "LAVA";
        data.color = RED;

        data.density = 2900.0;
        if (specificTempDetails) {
            data.thermalConductivity = 1.0; // W/m*K (approximate for molten rock)
            data.specificHeatCapacity = 1.5; // kJ/kg*K
        }
        data.temperature = 2050 + CELSIUS_TO_KELVIN;
        data.lowerTransitionPoint = 1000 + CELSIUS_TO_KELVIN;
        data.lowerTransitionType = ParticleType::STONE;
        data.upperTransitionPoint = 10000 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::PLASMA;
        data.state = ParticleState::FLUID;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::CLONE) {
        data.name = "CLONE";
        data.color = GOLD;

        data.density = 9999.9;
        data.thermalConductivity = 0; // Clone has no Thermal Conductivity
        data.specificHeatCapacity = 0; // Clone has no Thermal Density
        data.state = ParticleState::SOLID;
    }
    else if (type == ParticleType::ICE) {
        data.name = "ICE";
        data.color = SKYBLUE;

        data.density = 916.7;
        if (specificTempDetails) {
            data.thermalConductivity = 2.2; // Thermal conductivity of ice (W/m*K)
            data.specificHeatCapacity = 2.09; // Specific heat capacity of ice (kJ/kg*K)
        }
        data.temperature = -20 + CELSIUS_TO_KELVIN;
        data.upperTransitionPoint = 0 + CELSIUS_TO_KELVIN;
        data.upperTransitionType = ParticleType::WATER;
        data.state = ParticleState::SOLID;
    }
    else if (type == ParticleType::PLASMA) {
        data.name = "PLASMA";
        data.color = PURPLE;

        data.density = 0.02;
        if (specificTempDetails) {
            data.thermalConductivity = 0.1; // W/m*K (very rough approximation for low-density plasma)
            data.specificHeatCapacity = 5.0; // kJ/kg*K (varies widely with temperature and ionization state)
        }
        data.temperature = 9500 + CELSIUS_TO_KELVIN;
        data.lowerTransitionPoint = 3000 + CELSIUS_TO_KELVIN;
        data.lowerTransitionType = ParticleType::EMPTY;
        data.state = ParticleState::PLASMA;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::WALL) {
        data.name = "WALL";
        data.color = GRAY;

        data.density = 9999.9;
        data.thermalConductivity = 0; // Wall has no Thermal Conductivity
        data.specificHeatCapacity = 0; // Wall has no Thermal Density
        data.state = ParticleState::SOLID;
    }
    else if (type == ParticleType::DIAMOND) {
        data.name = "DIAMOND";
        data.color = glm::mix(BLUE, SKYBLUE, 0.5);

        data.density = 3500.0;
        if (specificTempDetails) {
            data.thermalConductivity = 1500.0; // W/m*K (very rough approximation for low-density plasma)
            data.specificHeatCapacity = 5.0; // kJ/kg*K (varies widely with temperature and ionization state)
        }
        data.state = ParticleState::SOLID;
    }
    else if (type == ParticleType::MERCURY) {
        data.name = "MERCURY";
        data.color = glm::mix(GRAY, WHITE, 0.5);

        // Physical properties for liquid mercury
        data.density = 13546.0; // kg/m³ (density of mercury at room temperature)
        if (specificTempDetails) {
            data.thermalConductivity = 8.3; // W/m*K (thermal conductivity of mercury at room temperature)
            data.specificHeatCapacity = 0.14; // kJ/kg*K (specific heat capacity of mercury at room temperature)
        }
        //data.lowerTransitionPoint = -38.83 + CELSIUS_TO_KELVIN; // Freezing point of mercury
        //data.upperTransitionPoint = 356.73 + CELSIUS_TO_KELVIN; // Boiling point of mercury
        //data.lowerTransitionType = ParticleType::SOLID_MERCURY; // Hypothetical solid state
        //data.upperTransitionType = ParticleType::GASEOUS_MERCURY; // Hypothetical gaseous state
        data.state = ParticleState::FLUID;
        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::OIL) {
        data.name = "OIL";
        data.color = glm::vec4(112.0 / 255.0, 22.0 / 255.0, 6.0 / 255.0, 1.0); // deep brown

        data.density = 870.0; // kg/m³ (density of crude oil, can vary based on type)
        if (specificTempDetails) {
            data.thermalConductivity = 0.13; // W/m*K (thermal conductivity of crude oil)
            data.specificHeatCapacity = 2.1; // kJ/kg*K (specific heat capacity of crude oil)
        }
        data.upperTransitionPoint = 300 + CELSIUS_TO_KELVIN; // Hypothetical boiling point (actual varies with type and conditions)
        data.upperTransitionType = ParticleType::FIRE; // Hypothetical gaseous state
        data.state = ParticleState::FLUID;

        AlchemicReaction reaction;
        reaction.prerequisites.push_back({ ParticleType::FIRE });
        reaction.results.push_back({ ParticleType::FIRE, 1200 + CELSIUS_TO_KELVIN });
        reaction.halflife = getRoughly(1.0 / 8.0, 0.1);
        data.reactions.push_back(reaction);

        data.movementDirections = getMovementDirectionsFromDensity(data.state, data.density);
    }
    else if (type == ParticleType::ERASER) {
        data.name = "ERASER";
        data.color = glm::mix(RED, BLACK, 0.5);

        data.density = 9999.9;
        data.thermalConductivity = 0; // Eraser has no Thermal Conductivity
        data.specificHeatCapacity = 0; // Eraser has no Thermal Density
        data.state = ParticleState::SOLID;
    }
    else if (type == ParticleType::WOOD) {
        data.name = "WOOD";
        data.color = glm::vec4(139.0 / 255.0, 69.0 / 255.0, 19.0 / 255.0, 1.0); // a lightish brown color

        data.density = 600.0; // kg/m³ (average density of wood, varies with moisture content and type)
        if (specificTempDetails) {
            data.thermalConductivity = 0.15; // W/m*K (thermal conductivity of wood)
            data.specificHeatCapacity = 1.7; // kJ/kg*K (specific heat capacity of wood)
        }
        data.upperTransitionPoint = 350 + CELSIUS_TO_KELVIN; // Approximate ignition temperature of wood
        data.upperTransitionType = ParticleType::BURNING_WOOD; // Turns to ash when combusted
        data.state = ParticleState::SOLID;

        AlchemicReaction reaction;
        reaction.prerequisites.push_back({ ParticleType::FIRE });
        reaction.results.push_back({ ParticleType::BURNING_WOOD, 500 + CELSIUS_TO_KELVIN });
        reaction.halflife = getRoughly(1.0 / 3.0, 0.1); // Wood is highly flamable
        data.reactions.push_back(reaction);

        reaction = {};
        reaction.prerequisites.push_back({ ParticleType::BURNING_WOOD });
        reaction.results.push_back({ ParticleType::BURNING_WOOD, 500 + CELSIUS_TO_KELVIN });
        reaction.halflife = getRoughly(1.0 / 300.0, 0.1); // fire spreads fairly quick in wood
        data.reactions.push_back(reaction);
    }
    else if (type == ParticleType::BURNING_WOOD) {
        data.name = "BURNING_WOOD";
        data.color = glm::mix(glm::vec4(139.0 / 255.0, 69.0 / 255.0, 19.0 / 255.0, 1.0), BLACK, 0.5); // a darkened, lightish brown color

        data.density = 600.0; // kg/m³ (average density of wood, varies with moisture content and type)
        if (specificTempDetails) {
            data.thermalConductivity = 0.15; // W/m*K (thermal conductivity of wood)
            data.specificHeatCapacity = 1.7; // kJ/kg*K (specific heat capacity of wood)
        }
        data.temperature = 500 + CELSIUS_TO_KELVIN;
        data.lowerTransitionPoint = 150 + CELSIUS_TO_KELVIN;
        data.lowerTransitionType = ParticleType::WOOD;
        data.upperTransitionPoint = 1000 + CELSIUS_TO_KELVIN; // Approximate ignition temperature of wood
        data.upperTransitionType = ParticleType::FIRE; // Turns to ash when combusted
        data.state = ParticleState::SOLID;

        data.emissions.push_back({ ParticleType::FIRE, getRoughly(1.0 / 5.0, 0.1) });

        AlchemicReaction reaction;
        reaction.prerequisites.push_back({ ParticleType::EMPTY });
        reaction.results.push_back({ ParticleType::FIRE, 950 + CELSIUS_TO_KELVIN }); // Turns to ash at a high temperature
        reaction.halflife = getRoughly(1.0 / 300.0, 0.1); // wood can burn a fairly long time before extinguishing
        data.reactions.push_back(reaction);

        reaction = {};
        reaction.prerequisites.push_back({ ParticleType::FIRE });
        reaction.results.push_back({ ParticleType::FIRE, 950 + CELSIUS_TO_KELVIN });
        reaction.halflife = getRoughly(1.0 / 300.0, 0.1); // wood can burn a fairly long time before extinguishing
        data.reactions.push_back(reaction);

        reaction = {};
        reaction.prerequisites.push_back({ ParticleType::WATER });
        reaction.results.push_back({ ParticleType::WOOD, -1 });
        reaction.halflife = getRoughly(1.0 / 3.0, 0.1); // water puts out fires quickly
        data.reactions.push_back(reaction);
    }

    data.color = glm::mix(data.color, BLACK, getRoughly(0.1, 1.0));
    data.density = getRoughly(data.density, 0.0001);

    //float printJ = data.density * data.specificHeatCapacity;
    //if (printJ != 0 && printJ != lastPrintJ) {
    //    std::cout << "J/C: " << printJ << std::endl;
    //    lastPrintJ = printJ;
    //}

    return data;
}

struct Particle;

// Create a grid to store particles
std::vector<std::vector<Particle>> grid(GRID_WIDTH, std::vector<Particle>(GRID_HEIGHT));

bool isValidIndex(int x, int y) {
    return (x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT);
}

// Define an enum for neighborhood types
enum class NeighborhoodType {
    Moore,
    Margolus
};

// Function to get neighbors using Moore neighborhood
std::vector<std::pair<int, int>> getMooreNeighbours(std::pair<int, int> pos, int radius) {
    std::vector<std::pair<int, int>> neighbors;

    // Moore neighborhood (square radius)
    const int maxNeighbors = (2 * radius + 1) * (2 * radius + 1) - 1;
    neighbors.reserve(maxNeighbors); // Preallocate memory

    int startX = pos.first - radius;
    int startY = pos.second - radius;
    int endX = pos.first + radius;
    int endY = pos.second + radius;

    for (int x = startX; x <= endX; ++x) {
        for (int y = startY; y <= endY; ++y) {
            // Skip the center point
            if (x == pos.first && y == pos.second) continue;

            // Check if the index is valid
            if (isValidIndex(x, y)) {
                neighbors.emplace_back(x, y);
            }
        }
    }

    return neighbors;
}

// Function to get neighbors using Margolus neighborhood
std::vector<std::pair<int, int>> getMargolusNeighbours(std::pair<int, int> pos) {
    std::vector<std::pair<int, int>> neighbors;

    // Margolus neighborhood (2x2 block)
    int blockX = (pos.first / 2) * 2;  // Align to the nearest even x
    int blockY = (pos.second / 2) * 2; // Align to the nearest even y

    // Depending on the position's offset within the block, define the neighborhood
    std::vector<std::pair<int, int>> blockOffsets = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    for (const auto& offset : blockOffsets) {
        int neighborX = blockX + offset.first;
        int neighborY = blockY + offset.second;

        // Skip the center point
        if (neighborX == pos.first && neighborY == pos.second) continue;

        // Check if the index is valid
        if (isValidIndex(neighborX, neighborY)) {
            neighbors.emplace_back(neighborX, neighborY);
        }
    }

    return neighbors;
}

// Define a structure for particles
struct Particle {
    generalParticleData data;

    std::vector<void (Particle::*)(std::pair<int, int>)> specialPreActions; // at the start of a frame before any particles have updated
    std::vector<void (Particle::*)(std::pair<int, int>)> specialActions; // any point in a frame after this particle has updated
    std::vector<void (Particle::*)(std::pair<int, int>)> specialPostActions; // at the end of a frame after all particles have updated

    Particle(ParticleType t = ParticleType::EMPTY) {

        data = getParticleData(t);

        switch (t) {
        case ParticleType::CLONE:
            specialActions.push_back(&Particle::clone);
            break;
        default:
            break;
        }

        // add halflife
        if (data.halflife != -1) {
            specialPostActions.push_back(&Particle::checkHalfLifeExpired);
        }

        // Add heat transfer function to special actions for particles that conduct heat
        if (data.thermalConductivity > 0 && data.specificHeatCapacity > 0) {
            if (data.lowerTransitionPoint != -1) {
                data.lowerTransitionPoint = getRoughly(data.lowerTransitionPoint, 0.01);
            }
            if (data.upperTransitionPoint != 9999999.9) {
                data.upperTransitionPoint = getRoughly(data.upperTransitionPoint, 0.01);
            }
            data.thermalConductivity = getRoughly(data.thermalConductivity, 0.01);
            data.specificHeatCapacity = getRoughly(data.specificHeatCapacity, 0.01);
            specialPreActions.push_back(&Particle::transferHeatFirstPass);
            specialPostActions.push_back(&Particle::transferHeatSecondPass);
        }

        // add alchemy
        if (!data.reactions.empty()) {
            specialPostActions.push_back(&Particle::checkAlchemyReactions);
        }

        // add particle emissions
        if (!data.emissions.empty()) {
            specialPostActions.push_back(&Particle::attemptEmissions);
        }
    }

    void transferParticleData(std::pair<int, int> pos, Particle newParticle, bool copySourceTemp = true) {
        generalParticleData dataCopy = grid[pos.first][pos.second].data;

        grid[pos.first][pos.second] = newParticle;

        if (copySourceTemp) {
            grid[pos.first][pos.second].data.temperature = dataCopy.temperature;
            //grid[pos.first][pos.second].data.temperature = std::max(grid[pos.first][pos.second].data.temperature, dataCopy.temperature);
        }
    }

    void performSpecialActions(std::vector<void (Particle::*)(std::pair<int, int>)>& _specialActions, std::pair<int, int> pos) {
        for (auto& action : _specialActions) {
            if (action) {
                (this->*action)(pos);  // Correct way to call a member function pointer
            }
        }
    }

    std::vector<std::pair<int, int>> getNeighbours(std::pair<int, int> pos, NeighborhoodType type = NeighborhoodType::Moore) {
        if (type == NeighborhoodType::Moore) {
            return getMooreNeighbours(pos, 1);
        }
        else if (type == NeighborhoodType::Margolus) {
            return getMargolusNeighbours(pos);
        }
        else {
            std::cerr << "Unknown neighborhood type!" << std::endl;
            return {};
        }
    }

    void checkAlchemyReactions(std::pair<int, int> pos) {
        std::vector<std::pair<int, int>> neighbors = getNeighbours(pos);

        for (AlchemicReaction reaction : data.reactions) {
            bool valid = false;

            if (RNG<float>::getRange(0, 1) < reaction.halflife) {
                valid = true;
            }

            for (AlchemicPrerequisites prerequisite : reaction.prerequisites) {
                int count = 0;
                for (std::pair<int, int> neighborPos : neighbors) {
                    Particle& current = grid[neighborPos.first][neighborPos.second];
                    if (current.data.type == prerequisite.type) {
                        count += 1;
                    }
                }
                if (count == 0) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                transferParticleData(pos, Particle(reaction.results[0].type));
                if (reaction.results[0].particleTemp != -1) {
                    grid[pos.first][pos.second].data.temperature = std::max(getRoughly(reaction.results[0].particleTemp, 0.1), grid[pos.first][pos.second].data.temperature);
                }
                break;
            }
        }
    }

    void attemptEmissions(std::pair<int, int> pos) {
        std::vector<std::pair<int, int>> neighbors = getNeighbours(pos);

        std::vector<std::pair<int, int>> emptyNeighbors;
        for (std::pair<int, int> neighborPos : neighbors) {
            Particle& neighbor = grid[neighborPos.first][neighborPos.second];
            if (neighbor.data.type == ParticleType::EMPTY) {
                emptyNeighbors.push_back(neighborPos);
            }
        }

        std::vector<Emission> emissions = data.emissions;

        std::shuffle(emissions.begin(), emissions.end(), RandomDevice::gen);

        for (Emission emission : emissions) {
            if (emptyNeighbors.empty()) {
                break;
            }

            if (RNG<double>::getRange(0, 1) < emission.halflife) {
                int randomIndex = rand() % emptyNeighbors.size();
                grid[emptyNeighbors[randomIndex].first][emptyNeighbors[randomIndex].second] = Particle(emission.type);
            }
        }
    }

    void checkHalfLifeExpired(std::pair<int, int> pos) {
        if (data.type != ParticleType::EMPTY) { // Check only non-empty particles
            // Handle particle decay or transformation
            if (data.halflife != -1) {
                if (RNG<double>::getRange(0, 1) < data.halflife) {
                    transferParticleData(pos, Particle(data.endOfLifeType));
                }
            }
        }
    }

    ParticleType rememberedParticleType = ParticleType::EMPTY;
    void clone(std::pair<int, int> pos) {
        std::vector<std::pair<int, int>> emptyNeighbors;

        std::vector<std::pair<int, int>> neighbors = getNeighbours(pos);
        for (std::pair<int, int> neighborPos : neighbors) {
            Particle& neighbor = grid[neighborPos.first][neighborPos.second];
            if (rememberedParticleType == ParticleType::EMPTY && neighbor.data.type != ParticleType::CLONE && neighbor.data.type != ParticleType::EMPTY) {
                rememberedParticleType = neighbor.data.type;
            }
            else if (neighbor.data.type == ParticleType::EMPTY) {
                emptyNeighbors.push_back(neighborPos);
            }
        }

        if (rememberedParticleType != ParticleType::EMPTY && !emptyNeighbors.empty()) {
            // Randomly pick an empty neighbor to clone into
            int randomIndex = rand() % emptyNeighbors.size();
            grid[emptyNeighbors[randomIndex].first][emptyNeighbors[randomIndex].second] = Particle(rememberedParticleType);
        }
    }

    void transferHeatFirstPass(std::pair<int, int> pos) {
        std::vector<std::pair<int, int>> neighbors = getNeighbours(pos);
        Particle& current = grid[pos.first][pos.second];
    
        int numNeighbors = neighbors.size();
    
        for (const std::pair<int, int>& neighborPos : neighbors) {
            Particle& neighbor = grid[neighborPos.first][neighborPos.second];
    
            if (neighbor.data.thermalConductivity > 0.0f && neighbor.data.specificHeatCapacity > 0.0f) {
                double tempDelta = current.data.temperature - neighbor.data.temperature;
    
                // Calculate heat transfer considering both particles' conductivities
                //float combinedConductivity = (current.data.thermalConductivity + neighbor.data.thermalConductivity) * 0.5f;
                double combinedConductivity = std::min(current.data.thermalConductivity, neighbor.data.thermalConductivity);
                double heatTransfer = combinedConductivity * tempDelta;
    
                // Calculate the heat exchange considering thermal densities
                double totalDensity = current.data.specificHeatCapacity + neighbor.data.specificHeatCapacity;
                if (totalDensity > 0.0f) {
                    // Normalize the heat exchange by the number of neighbors
                    double heatExchange = (0.5f * heatTransfer / totalDensity) / numNeighbors;
    
                    // Store the heat to be transferred, ensuring conservation
                    current.data.heatReceived -= heatExchange * (neighbor.data.specificHeatCapacity / current.data.specificHeatCapacity);
                    neighbor.data.heatReceived += heatExchange * (current.data.specificHeatCapacity / neighbor.data.specificHeatCapacity);
                }
            }
        }
    }
    
    void transferHeatSecondPass(std::pair<int, int> pos) {
        Particle& current = grid[pos.first][pos.second];
    
        // Apply the heat received from neighbors and reset
        current.data.temperature += current.data.heatReceived;
        current.data.heatReceived = 0.0f; // Reset after applying to avoid accumulation
    
        // Check for phase transitions based on the updated temperature
        if (data.lowerTransitionPoint != -1) {
            if (data.temperature < data.lowerTransitionPoint) {
                transferParticleData(pos, Particle(data.lowerTransitionType));
            }
        }
    
        if (data.upperTransitionPoint != 9999999.9) {
            if (data.temperature > data.upperTransitionPoint) {
                transferParticleData(pos, Particle(data.upperTransitionType));
            }
        }
    }
};

void setWalls(ParticleType type) {
    // Set top and bottom walls
    for (int x = 0; x < GRID_WIDTH; x++) {
        grid[x][0] = Particle(type); // Top wall
        grid[x][GRID_HEIGHT - 1] = Particle(type); // Bottom wall
    }

    // Set left and right walls
    for (int y = 0; y < GRID_HEIGHT; y++) {
        grid[0][y] = Particle(type); // Left wall
        grid[GRID_WIDTH - 1][y] = Particle(type); // Right wall
    }
}

void InitializeGrid() {
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            grid[x][y] = Particle(ParticleType::EMPTY);
        }
    }
}

// Create a list of all positions in the grid
std::vector<std::pair<int, int>> positions;

//void fluidJank() {
//    // adds vertical fluid movement to allow leveling under barriers
//    if (particle.data.state == ParticleState::FLUID) {
//        std::vector<int> offsets = { -1, 0, 1 };
//        std::shuffle(offsets.begin(), offsets.end(), RandomDevice::gen);
//    
//        int totalFluidNeighbours = 0;
//        for (int offset_x : offsets) {
//            if (isValidIndex(pos.first + offset_x, pos.second - 1)) {
//                if (grid[pos.first + offset_x][pos.second - 1].data.type == particle.data.type) {
//                    totalFluidNeighbours += 1;
//                }
//            }
//        }
//    
//        if (totalFluidNeighbours == 3) {
//            if (RNG<double>::getRange(0, 1) < 0.9) {
//                for (int offset_x : offsets) {
//                    if (isValidIndex(pos.first + offset_x, pos.second + 1)) {
//                        if (grid[pos.first + offset_x][pos.second + 1].data.type == ParticleType::EMPTY) {
//                            if (StepInDirection({ pos.first, pos.second }, firstPos, particle, { offset_x, 1 }, depth + 1, timesSwapped)) {
//                                return true; // Exit after first successful move
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

int maxStepDepth = 8;
bool StepInDirection(std::pair<int, int> pos, std::pair<int, int> firstPos, Particle& particle, std::pair<int, int> direction, int depth = 0, int timesSwapped = 0) {
    int x = firstPos.first;
    int y = firstPos.second;

    int newX = pos.first + direction.first;
    int newY = pos.second + direction.second;

    if (depth - 1 >= maxStepDepth) {
        return false;
    }

    //fluidJank();

    // Check if the new position is within bounds
    if (isValidIndex(newX, newY)) {
        if (grid[newX][newY].data.type == ParticleType::EMPTY) { // Changed from `grid[newX][newY].data.type`
            if (timesSwapped == 0) {
                grid[newX][newY] = std::move(particle);
                grid[x][y] = Particle(ParticleType::EMPTY); // Clear the previous position
                return true; // Exit after first successful move
            }
        }
        else if (grid[newX][newY].data.type == ParticleType::ERASER) {
            grid[x][y] = Particle(ParticleType::EMPTY); // Clear the previous position
            return true; // Exit after first successful move
        }
        // If no movement was possible, try swapping based on density
        else if (!grid[newX][newY].data.movementDirections.empty()) {//if (particle.type != grid[newX][newY].type) {
            if (particle.data.state == ParticleState::FLUID && direction.first != 0) {
                if (grid[newX][newY].data.state != ParticleState::FLUID) {
                    timesSwapped += 1;
                }
                if (StepInDirection({ newX, newY }, firstPos, particle, { direction.first, 0 }, depth + 1, timesSwapped)) {
                    return true; // Exit after first successful move
                }
            }

            if (depth != 0) {// && particle.data.state != ParticleState::FLUID) { // only attempt swaps on the first move unless particle is a fluid
                return false;
            }

            double currentDensity = particle.data.density;
            double neighborDensity = grid[newX][newY].data.density;

            // Ensure densities are not zero to avoid division by zero
            if (currentDensity > 0.0f && neighborDensity > 0.0f && currentDensity != neighborDensity) {
                // Determine the larger and smaller densities
                double largerDensity = std::max(currentDensity, neighborDensity);
                double smallerDensity = std::min(currentDensity, neighborDensity);

                // Calculate the swap probability proportional to the density difference
                double swapProbability = smallerDensity / largerDensity;

                swapProbability = (currentDensity < 1.2 ? swapProbability : 1.0 - swapProbability);

                bool densityDirectionCheck = (currentDensity < 1.2 ? (currentDensity == smallerDensity) : (currentDensity == largerDensity));
                if (RNG<double>::getRange(0, 1) < swapProbability && // Lower probability for closer densities
                    densityDirectionCheck) {
                    // Swap particles to new positions
                    std::swap(grid[x][y], grid[newX][newY]);
                    return true; // Exit after first successful swap
                }
            }
        }
    }

    return false;
}

void MoveParticle(std::pair<int, int> pos, Particle& particle) {
    int x = pos.first;
    int y = pos.second;

    bool moved = false;
    const auto& movementDirections = particle.data.movementDirections; // Keep const reference
    std::vector<size_t> tierIndices(movementDirections.size());
    std::iota(tierIndices.begin(), tierIndices.end(), 0); // Initialize indices for random access

    while (!moved && !tierIndices.empty()) {
        // Calculate probabilities based on the weights of the remaining tiers
        std::vector<float> weights;
        weights.reserve(tierIndices.size());
        for (size_t index : tierIndices) {
            weights.push_back(movementDirections[index].first);
        }

        // Sample a tier based on probabilities
        int chosenTierIndex = sampleFromProbabilities(weights);
        size_t actualTierIndex = tierIndices[chosenTierIndex];

        // Randomly sample directions without shuffling
        auto& directions = movementDirections[actualTierIndex].second;
        std::vector<size_t> directionIndices(directions.size());
        std::iota(directionIndices.begin(), directionIndices.end(), 0); // Indices for random access

        // Randomly pick a direction index
        std::shuffle(directionIndices.begin(), directionIndices.end(), RandomDevice::gen);

        for (size_t dirIndex : directionIndices) {
            const auto& direction = directions[dirIndex];

            moved = StepInDirection(pos, pos, particle, direction);
            if (moved) {
                break;
            }
        }

        if (!moved) {
            // Remove the chosen tier index if no movement occurred
            tierIndices.erase(tierIndices.begin() + chosenTierIndex);
        }
    }
}

void UpdateParticles() {
    // Shuffle the list of positions to randomize the update order
    std::shuffle(positions.begin(), positions.end(), RandomDevice::gen);

    // At the start of each frame, perform all pre-frame special actions
    for (const auto& pos : positions) {
        int x = pos.first;
        int y = pos.second;
        Particle& particle = grid[x][y];

        particle.performSpecialActions(particle.specialPreActions, pos);
    }

    // During each frame, perform all normal special actions
    for (const auto& pos : positions) {
        int x = pos.first;
        int y = pos.second;
        Particle& particle = grid[x][y];

        if (particle.data.type != ParticleType::EMPTY) { // Check only non-empty particles
            MoveParticle(pos, particle);
        }

        particle.performSpecialActions(particle.specialActions, pos);
    }

    // At the end of each frame, perform all post-frame special actions
    for (const auto& pos : positions) {
        int x = pos.first;
        int y = pos.second;
        Particle& particle = grid[x][y];

        particle.performSpecialActions(particle.specialPostActions, pos);
    }
}

wrapValue selected(0, int(ParticleType::COUNT) - 2);
Text selectedThing;
Text hoveredThing;
Text generalInfoBox;

int numParticles = 0;
void getCellInfo(Particle& particle) {
    std::string infoString;

    infoString += particle.data.name;

    if (particle.data.thermalConductivity > 0.0 && particle.data.specificHeatCapacity > 0.0) {
        infoString += ", Temp: " + to_string_rounded(particle.data.temperature - CELSIUS_TO_KELVIN, 2) + "C";
    }

    if (particle.data.density != 0.0) {
        infoString += ", Density: " + to_string_rounded(particle.data.density, 3);
    }

    hoveredThing.setString(infoString);
}

void getGeneralInfo() {
    std::string infoString;

    infoString += "Total Particles: " + std::to_string(numParticles);
    infoString += "    FPS: " + std::to_string(GetFps());

    generalInfoBox.setString(infoString);
}

int brushRadius = 0;
bool paused = false;
bool playOneFrame = false;
void PollCustomEvents2(Camera2D cam) {
    if (IsKeyPressed(GLFW_KEY_SPACE)) {
        paused = !paused;
    }

    bool selectedChanged = false;

    int scroll = GetMouseWheelMove();
    if (scroll != 0) {
        if (IsKeyDown(GLFW_KEY_LEFT_SHIFT)) {
            brushRadius += scroll;
            brushRadius = std::max(0, brushRadius);
        }
        else {
            selected -= scroll;
            selectedChanged = true;
        }
    }

    int mouseX = GetMouseX(cam) / CELL_SIZE;
    int mouseY = GetMouseY(cam) / CELL_SIZE;
    for (int x = -brushRadius; x < brushRadius + 1; x++) {
        for (int y = -brushRadius; y < brushRadius + 1; y++) {
            if (isValidIndex(mouseX + x, mouseY + y)) {
                if (IsMouseButtonDown(GLFW_MOUSE_BUTTON_1)) { // Place stuff with left mouse click
                    if (grid[mouseX + x][mouseY + y].data.type == ParticleType::EMPTY) {
                        grid[mouseX + x][mouseY + y] = Particle(ParticleType(selected.value + 1));
                    }
                }
                else if (IsMouseButtonDown(GLFW_MOUSE_BUTTON_2)) { // Erase with right click
                    grid[mouseX + x][mouseY + y] = Particle(ParticleType::EMPTY);
                }

                if (x == 0 && y == 0) {
                    if (IsMouseButtonPressed(GLFW_MOUSE_BUTTON_3)) {
                        if (grid[mouseX + x][mouseY + y].data.type != ParticleType::EMPTY) {
                            selected = int(grid[mouseX + x][mouseY + y].data.type) - 1;
                            selectedChanged = true;
                        }
                    }
                    getCellInfo(grid[mouseX][mouseY]);
                }
            }
        }
    }

    getGeneralInfo();

    if (IsKeyPressed(GLFW_KEY_W)) {
        setWalls(ParticleType::WALL);
    }
    else if (IsKeyPressed(GLFW_KEY_E)) {
        setWalls(ParticleType::ERASER);
    }
    else if (IsKeyPressed(GLFW_KEY_C)) {
        InitializeGrid();
    }

    if (IsKeyPressed(GLFW_KEY_F)) {
        paused = true;
        playOneFrame = true;
    }

    if (selectedChanged) {
        generalParticleData data = getParticleData(ParticleType(selected.value + 1));
        selectedThing.setString(data.name);
        selectedThing.setColor(data.color);
    }
}

const char* ParticleVertexShaderSource = R"glsl(
    #version 330 core

    layout(location = 0) in vec3 aPos;       // Vertex position
    layout(location = 1) in vec2 aTexCoord;  // Texture coordinates
    layout(location = 2) in vec4 aColor;     // Vertex color

    out vec2 TexCoord;    // Passed to fragment shader
    out vec4 VertexColor; // Passed to fragment shader

    uniform mat4 projection; // Projection matrix (optional, if you have camera perspective)
    uniform mat4 view;       // View matrix (optional, if you have camera perspective)

    void main() {
        // Apply transformations
        gl_Position = projection * view * vec4(aPos, 1.0);

        // Pass texture coordinates and color to fragment shader
        TexCoord = aTexCoord;
        VertexColor = aColor;
    }
)glsl";

const char* ParticleFragmentShaderSource = R"glsl(
    #version 330 core

    in vec2 TexCoord;     // From vertex shader
    in vec4 VertexColor;  // From vertex shader

    out vec4 FragColor;   // Final output color

    uniform sampler2D texture1; // Texture sampler
    uniform bool useTexture;    // Whether to use the texture or just vertex color

    void main() {
        if (useTexture) {
            // Mix texture color with vertex color
            FragColor = texture(texture1, TexCoord) * VertexColor;
        }
        else {
            // Use only vertex color
            FragColor = VertexColor;
        }
    }
)glsl";


struct Vertex {
    glm::vec3 position;
    glm::vec2 texCoords;
    glm::vec4 color;
};

// Global batch storage
std::vector<Vertex> batchVertices;
GLuint batchVBO, batchVAO;

// Shader instance
Shader particleShader("ParticleShader");

// Setup function to initialize VBO and VAO for batching
void SetupBatchRendering() {
    glGenVertexArrays(1, &batchVAO);
    glGenBuffers(1, &batchVBO);

    glBindVertexArray(batchVAO);
    glBindBuffer(GL_ARRAY_BUFFER, batchVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * 10000, nullptr, GL_DYNAMIC_DRAW); // Allocate enough space initially

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    glEnableVertexAttribArray(1);

    // Color attribute
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Initialize shader
    if (!particleShader.loadFromFile(ParticleVertexShaderSource, ParticleFragmentShaderSource)) {
        std::cerr << "Failed to load particle shaders!" << std::endl;
        return;
    }

    // Compile and link shaders
    particleShader.end();
}

// Function to batch draw rectangles
void BatchDrawRectangle(float posX, float posY, float width, float height, float angle, glm::vec4* color = nullptr, Texture* texture = nullptr, glm::vec4* customRect = nullptr) {
    glm::vec4 vertexColor = (color != nullptr) ? *color : glm::vec4(1.0f); // Default to white if no color provided

    float left = 0.0f;
    float right = 1.0f;
    float bottom = 0.0f;
    float top = 1.0f;

    if (customRect != nullptr && texture != nullptr) {
        left = customRect->x / float(texture->width);
        right = (customRect->x + customRect->z) / float(texture->width);
        bottom = customRect->y / float(texture->height);
        top = (customRect->y + customRect->w) / float(texture->height);
    }

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(posX, posY, 0.0f));
    model = glm::rotate(model, angle, glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, glm::vec3(width, height, 1.0f));

    glm::vec3 topLeft = glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
    glm::vec3 bottomLeft = glm::vec3(model * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    glm::vec3 bottomRight = glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    glm::vec3 topRight = glm::vec3(model * glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));

    batchVertices.push_back({ topLeft, glm::vec2(left, top), vertexColor });
    batchVertices.push_back({ bottomLeft, glm::vec2(left, bottom), vertexColor });
    batchVertices.push_back({ bottomRight, glm::vec2(right, bottom), vertexColor });
    batchVertices.push_back({ topRight, glm::vec2(right, top), vertexColor });
}

// Function to execute the batch draw
void ExecuteBatchDraw(Texture* texture = nullptr) {
    if (!batchVertices.empty()) {
        // Use the shader program
        BeginShaderMode(particleShader);

        glBindVertexArray(batchVAO);
        glBindBuffer(GL_ARRAY_BUFFER, batchVBO);

        // Update VBO with new vertex data
        glBufferData(GL_ARRAY_BUFFER, batchVertices.size() * sizeof(Vertex), batchVertices.data(), GL_DYNAMIC_DRAW);

        particleShader.setUniform("projection", extras::activeCamera2D->GetProjectionMatrix());
        particleShader.setUniform("view", extras::activeCamera2D->GetViewMatrix());

        // Bind the texture if available
        if (texture != nullptr) {
            texture->bind(0);
            particleShader.setUniform("useTexture", 1);
        }
        else {
            particleShader.setUniform("useTexture", 0);
        }

        // Draw the batch
        glDrawArrays(GL_QUADS, 0, batchVertices.size());

        // Unbind the texture
        if (texture != nullptr) {
            texture->unbind();
        }

        // Clear batch vertices for the next frame
        batchVertices.clear();
        glBindVertexArray(0);
        EndShaderMode();
    }
}

// Function to render particles using batching
void RenderParticles() {
    numParticles = 0;
    // Start batch rendering
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            Particle& particle = grid[x][y];
            if (particle.data.type != ParticleType::EMPTY) {
                BatchDrawRectangle(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, 0.0f, &particle.data.color);
                numParticles++;
            }
        }
    }

    // Execute batch draw for all rectangles
    ExecuteBatchDraw();
}

int main(void)
{
    RandomDevice::reseed(0);
    InitWindow(GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE, "Fully Fledged Engine v0.0");
    InitializeGrid();

    SetupBatchRendering();

    SetTargetFPS(300);

    Camera2D cam;

    selected = 0;
    selectedThing = Text(*extras::defaultFont, "SAND", 16);
    selectedThing.setColor(YELLOW);
    selectedThing.background = true;

    hoveredThing = Text(*extras::defaultFont, "", 16);
    hoveredThing.background = true;

    generalInfoBox = Text(*extras::defaultFont, "", 16);
    generalInfoBox.background = true;

    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            positions.emplace_back(x, y);
        }
    }

    while (!WindowShouldClose()) {
        PollCustomEvents();
        float dt = GetFrameTime();

        if (!paused || playOneFrame) {
            UpdateParticles();
            playOneFrame = false;
        }

        PollCustomEvents2(cam);

        BeginDrawing();
        BeginMode2D(cam);
        ClearBackground(BLACK);

        RenderParticles();

        // top left text
        hoveredThing.Draw(5, cam.currentViewportSize.y - 5, false, false, true, true);
        generalInfoBox.Draw(5, cam.currentViewportSize.y - 35, false, false, true, true);

        // top right text
        selectedThing.Draw(cam.currentViewportSize.x - 5, cam.currentViewportSize.y - 5, false, false, false, true);

        EndMode2D();
        EndDrawing();
        std::string updatedTitle = "Fully Fledged Engine v0.0 | FPS: " + std::to_string(extras::perfLogger.GetFps());
        glfwSetWindowTitle(extras::ActiveWindow, updatedTitle.c_str());
    }

    CloseWindow();
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int nCmdShow) {
    return main();
}