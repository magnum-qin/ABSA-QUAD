"""Check diagnostic scoring boundaries without model weights or training."""
import unittest

from scripts.icsai.revision_diagnostics import implicit, label_counts, label_metrics, spans


class RevisionDiagnosticTests(unittest.TestCase):
    def test_orphan_inside_and_type_switch_match_greedy_contract(self):
        self.assertEqual(spans([2, 1, 2, 4, 3, 4, 0]),
                         {(1, 2, 'aspect'), (4, 5, 'opinion')})

    def test_boundary_error_is_not_a_span_match(self):
        metrics = label_metrics(label_counts([0, 1, 2, 0], [0, 1, 0, 0]))
        self.assertEqual(metrics['span_f1'], 0)
        self.assertAlmostEqual(metrics['bio_non_o_micro_f1'], 2/3)
        self.assertEqual(metrics['bio_accuracy'], 3/4)

    def test_cls_anchor_counts_as_a_supervised_span(self):
        self.assertEqual(spans([1, 0, 3, 0]), {(0, 0, 'aspect'), (2, 2, 'opinion')})
        metrics = label_metrics(label_counts([1, 0, 3, 0], [1, 0, 3, 0]))
        self.assertEqual(metrics['span_f1'], 1)

    def test_null_partition_uses_both_fields_and_official_lowercase(self):
        self.assertTrue(implicit({'Aspect': 'null', 'Opinion': 'nice'}))
        self.assertTrue(implicit({'Aspect': 'food', 'Opinion': 'NULL'}))
        self.assertFalse(implicit({'Aspect': 'food', 'Opinion': 'nice'}))

    def test_no_positive_predictions_has_zero_not_undefined_f1(self):
        self.assertEqual(label_metrics(label_counts([0, 0], [0, 0]))['span_f1'], 0)


if __name__ == '__main__':
    unittest.main()
