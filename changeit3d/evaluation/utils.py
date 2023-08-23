from ..analysis.most_frequent_expressions import frequent_expressions
from ..in_out.changeit3d_net import shape_with_expression_dataloader_convenient


# Recall. test geometry
#           => not seen by the 3D-AE
#           => not seen as target in the listener
#           => not manipulated by the editor (i.e. being a distractor) so as to beat/improve its compatibility with any sentence


def common_expressions_dataloader(shape_class, data_loader, vocab, batch_size=4096):

    expressions = frequent_expressions[shape_class]

    stimulus_index = 0
    expression_loader = \
        shape_with_expression_dataloader_convenient(data_loader,
                                                    expressions,
                                                    vocab,
                                                    batch_size=batch_size,
                                                    shape_class=shape_class,
                                                    num_workers=10,
                                                    verbose=True)

    return expression_loader, stimulus_index
